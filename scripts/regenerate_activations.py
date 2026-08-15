"""Re-add the `activation_vector` column to an NLA parquet that ships text only.

Some published NLA datasets (e.g. `ceselder/qwen3-8b-nla-L24-finefineweb-100k`)
store `detokenized_text_truncated` but drop the vectors — they are ~16 GB of
float32 and are exactly reproducible from the text. Every EasyNLA trainer
(`train_sft.py`, `train_rl_vllm.py`, `train_rl_self_contained.py`) reads
`schema.ACTIVATION_COLUMN`, so such a parquet cannot be trained on until the
column is rebuilt.

The contract (dataset card + `datagen/stage0_extract.py`): the activation is the
residual stream at `activation_layer`, at the FINAL token of
`detokenized_text_truncated` — the text is truncated to end exactly there, so
`len(tokenize(text)) == n_raw_tokens` and the position is `n_raw_tokens - 1`.

That identity holds for ~99.97% of rows. It fails where the extraction token was
a partial byte of a multibyte character: detokenizing emits U+FFFD and the text
no longer retokenizes to the original tokens. Those rows are DROPPED, not
repositioned — a vector taken from a silently different prefix is worse than a
missing row at this rate. A mismatch rate above `--max-drop-frac` aborts instead,
since that signals real tokenizer drift (wrong base model / tokenizer revision),
which would corrupt every vector rather than a handful.

Vectors are written RAW / unnormalized (`norm: none`), matching stage 0 —
normalization is a training-time decision driven by the sidecar's
`injection_scale` / `mse_scale`.

Data-parallel across GPUs: one full model replica per rank (8B bf16 ≈ 16 GB), a
contiguous row shard each, shard parquets merged back in original row order.
This is ~N× faster than `HFExtractor`'s `device_map="auto"`, which shards one
model across GPUs and leaves all but one idle.

    python scripts/regenerate_activations.py \
        --in data/av_sft_shuf.parquet --out data/av_sft_shuf.full.parquet \
        --base-model Qwen/Qwen3-8B
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

from nla.datagen._common import load_tokenizer
from nla.schema import ACTIVATION_COLUMN
from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_config

TEXT_COL = "detokenized_text_truncated"
NTOK_COL = "n_raw_tokens"
LAYER_COL = "activation_layer"


class _CaptureComplete(Exception):
    """Abort the forward once the target layer is captured — skips the upper
    decoder blocks and the lm_head, whose [B, seq, vocab] logits dominate both
    time and memory here. Same trick as datagen's HFExtractor."""


def _shard_bounds(n_rows: int, world: int) -> list[tuple[int, int]]:
    edges = [round(i * n_rows / world) for i in range(world + 1)]
    return list(zip(edges[:-1], edges[1:]))


def _keep_indices(args: argparse.Namespace, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices to extract, plus the token length of each kept row.

    One cheap CPU pass with the fast (Rust, multithreaded) tokenizer, so the GPU
    workers never see a row that would land on the wrong position.

    Two modes, because `n_raw_tokens` is only meaningful under the tokenizer that
    wrote it:

    - default (same-model regen): a row is kept iff its text retokenizes to
      exactly `n_raw_tokens`. See the module docstring for why the failures are
      dropped rather than repositioned.
    - `--cross-model` (retarget onto a DIFFERENT base model): `n_raw_tokens` came
      from the source model's tokenizer and will disagree on nearly every row, so
      there is nothing to validate against — it is RECOMPUTED here instead. This
      is not a weakening of the check: the contract is "the activation at the
      final token of `detokenized_text_truncated`", which is well-defined for any
      tokenizer, and the recomputed lengths are what the workers then assert
      against. The rows the same-model path drops (text ending mid-multibyte, so
      it no longer round-trips) are kept here — under a different tokenizer that
      text is simply text, and its final token is as valid as any other.
    """
    tok = load_tokenizer(args.base_model)
    tbl = pq.ParquetFile(args.inp).read(columns=[TEXT_COL, NTOK_COL]).slice(0, n_rows)
    texts = tbl.column(TEXT_COL).to_pylist()
    ntoks = tbl.column(NTOK_COL).to_numpy()

    got = np.empty(n_rows, dtype=np.int64)
    step = 20_000
    for s in range(0, n_rows, step):
        enc = tok(texts[s : s + step], add_special_tokens=True)["input_ids"]
        got[s : s + len(enc)] = [len(e) for e in enc]
    got = np.minimum(got, args.max_length)

    if args.cross_model:
        keep = np.arange(n_rows)
        src_med, new_med = int(np.median(ntoks)), int(np.median(got))
        n_repl = sum(1 for t in texts if "�" in t)
        print(
            f"cross-model retarget: keeping all {n_rows} rows; n_raw_tokens "
            f"RECOMPUTED under {args.base_model}'s tokenizer "
            f"(median {src_med} -> {new_med} tokens/row, "
            f"{n_repl} rows contain U+FFFD from the source truncation)",
            flush=True,
        )
        return keep, got

    keep = np.nonzero(got == np.minimum(ntoks, args.max_length))[0]
    dropped = n_rows - len(keep)
    frac = dropped / max(n_rows, 1)
    print(
        f"round-trip check: keeping {len(keep)}/{n_rows} rows "
        f"(dropped {dropped}, {frac:.4%} — text ends mid-multibyte-character)",
        flush=True,
    )
    assert frac <= args.max_drop_frac, (
        f"{frac:.2%} of rows fail the retokenize check (limit {args.max_drop_frac:.2%}). "
        f"That is tokenizer drift, not multibyte edge cases — check that "
        f"--base-model {args.base_model} is the model datagen actually used. "
        f"If you are deliberately retargeting this dataset onto a different base "
        f"model, pass --cross-model."
    )
    return keep, got[keep]


@torch.no_grad()
def _run_shard(rank: int, args: argparse.Namespace, bounds: list[tuple[int, int]]) -> None:
    lo, hi = bounds[rank]
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tok = load_tokenizer(args.base_model)
    # Right padding + right truncation: we index the final *real* token via the
    # attention mask, and left-truncation would silently change which token that is.
    tok.padding_side = "right"
    tok.truncation_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to(device)
    d_model = resolve_text_config(model.config).hidden_size

    # Only the rows that survived the round-trip check, in original order.
    keep = np.load(Path(args.shard_dir) / "keep_idx.npy")[lo:hi]
    pf = pq.ParquetFile(args.inp)
    tbl = pf.read(columns=[TEXT_COL, NTOK_COL, LAYER_COL]).take(pa.array(keep))
    texts = tbl.column(TEXT_COL).to_pylist()
    # Lengths come from the parent's tokenizer pass (already clamped to
    # max_length), not from the NTOK_COL column: under --cross-model the column
    # holds the SOURCE model's counts and is wrong for this tokenizer. In the
    # same-model path the two are equal by construction — every surviving row
    # round-tripped — so this is one code path for both modes.
    ntoks = np.load(Path(args.shard_dir) / "ntoks_kept.npy")[lo:hi]
    if args.layer is not None:
        layer_index = args.layer
    else:
        layers = np.unique(tbl.column(LAYER_COL).to_numpy())
        assert len(layers) == 1, f"mixed activation_layer values in shard: {layers}"
        layer_index = int(layers[0])

    captured: list[torch.Tensor | None] = [None]

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[0] = h.detach()
        raise _CaptureComplete

    handle = resolve_decoder_layers(model)[layer_index].register_forward_hook(hook)

    # Length-sorted batching: the shard spans 51..4096 tokens, so batching in
    # row order would pad short rows out to the longest in the batch and waste
    # most of the compute. Sort by length, batch to a token budget, then invert
    # the permutation before writing so row order is preserved exactly.
    order = np.argsort(ntoks, kind="stable")
    out = np.zeros((len(texts), d_model), dtype=np.float32)

    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in order:
        width = max(cur_max, min(int(ntoks[idx]), args.max_length))
        if cur and width * (len(cur) + 1) > args.batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            width = min(int(ntoks[idx]), args.max_length)
        cur.append(int(idx))
        cur_max = width
    if cur:
        batches.append(cur)

    t0 = time.time()
    done_tokens = 0
    for bi, batch in enumerate(batches):
        enc = tok(
            [texts[i] for i in batch],
            return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_length, add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attn = enc["attention_mask"].to(device, non_blocking=True)

        lengths = attn.sum(dim=1)
        # Guard, not a filter: main() already resolved the expected length of
        # every row (dropping mismatches in the same-model path, recomputing them
        # under --cross-model). Tripping here means the parent's tokenizer and
        # the worker's disagree.
        expect = torch.tensor([int(ntoks[i]) for i in batch], device=device)
        assert torch.equal(lengths, expect), (
            f"rank{rank} batch{bi}: retokenized length != n_raw_tokens "
            f"(first mismatch {int((lengths != expect).nonzero()[0][0])}) "
            f"despite the main-process filter — tokenizer nondeterminism?"
        )

        captured[0] = None
        try:
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        except _CaptureComplete:
            pass
        assert captured[0] is not None, (
            f"forward hook on decoder layer {layer_index} never fired"
        )
        h = captured[0]
        assert h.shape[-1] == d_model, f"captured width {h.shape[-1]} != d_model {d_model}"

        final = h[torch.arange(h.shape[0], device=device), lengths - 1]
        out[batch] = final.float().cpu().numpy()

        done_tokens += int(attn.sum())
        if rank == 0 and (bi % 50 == 0 or bi == len(batches) - 1):
            frac = (bi + 1) / len(batches)
            el = time.time() - t0
            print(
                f"[rank0] batch {bi+1}/{len(batches)} ({frac:6.1%})  "
                f"{done_tokens/max(el,1e-9)/1e3:.1f}k tok/s/gpu  "
                f"eta {el/max(frac,1e-9)*(1-frac)/60:.1f} min",
                flush=True,
            )
    handle.remove()

    shard = Path(args.shard_dir) / f"shard_{rank:03d}.parquet"
    pq.write_table(
        pa.table({ACTIVATION_COLUMN: pa.FixedSizeListArray.from_arrays(
            pa.array(out.reshape(-1), type=pa.float32()), d_model)}),
        shard, compression="none",
    )
    print(f"[rank{rank}] wrote {shard} rows={len(texts)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--batch-tokens", type=int, default=131072,
                   help="padded tokens per forward batch, per GPU")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
    p.add_argument("--max-drop-frac", type=float, default=0.01,
                   help="abort if more than this fraction fails the retokenize check")
    p.add_argument("--cross-model", action="store_true",
                   help="Retarget onto a DIFFERENT base model than the one that "
                        "wrote the dataset (e.g. Qwen-tokenized text -> Gemma "
                        "activations). Skips the n_raw_tokens round-trip check, "
                        "which can only hold for the original tokenizer, and "
                        "RECOMPUTES that column under --base-model's tokenizer. "
                        "The extracted position is unchanged in meaning: the "
                        "final token of detokenized_text_truncated. Pair it with "
                        "--layer, since a new model's depths are its own.")
    p.add_argument("--layer", type=int, default=None,
                   help="Extraction layer. Default: the dataset's own activation_layer. "
                        "Set this to RETARGET a dataset at a different model/depth — "
                        "the output's activation_layer column and sidecar are rewritten "
                        "to match, so the result is self-describing. layer=K means the "
                        "OUTPUT of block K (== HF hidden_states[K+1]), matching "
                        "datagen/extractors.py and models.py.")
    p.add_argument("--shard-dir", default="")
    p.add_argument("--keep-shards", action="store_true")
    args = p.parse_args()

    src = pq.ParquetFile(args.inp)
    n_rows = src.metadata.num_rows
    if args.limit:
        n_rows = min(n_rows, args.limit)
    assert ACTIVATION_COLUMN not in src.schema_arrow.names, (
        f"{args.inp} already has {ACTIVATION_COLUMN} — nothing to regenerate"
    )

    # Resolved here too (the workers derive them independently) so the sidecar
    # written below describes the run without depending on worker internals.
    d_model = resolve_text_config(AutoConfig.from_pretrained(args.base_model)).hidden_size
    layer_index = args.layer
    if layer_index is None:
        _l = np.unique(src.read(columns=[LAYER_COL]).column(0).to_numpy())
        assert len(_l) == 1, f"mixed {LAYER_COL} values in {args.inp}: {_l}"
        layer_index = int(_l[0])
    n_layers = resolve_text_config(
        AutoConfig.from_pretrained(args.base_model)).num_hidden_layers
    assert 0 <= layer_index < n_layers, (
        f"--layer {layer_index} out of range for {args.base_model} "
        f"({n_layers} blocks)")
    print(f"{args.base_model}: {n_layers} blocks, d_model={d_model} | "
          f"extracting the OUTPUT of block {layer_index}")

    args.shard_dir = args.shard_dir or f"{args.out}.shards"
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)

    keep_idx, ntoks_kept = _keep_indices(args, n_rows)
    np.save(Path(args.shard_dir) / "keep_idx.npy", keep_idx)
    np.save(Path(args.shard_dir) / "ntoks_kept.npy", ntoks_kept)
    n_keep = len(keep_idx)
    assert len(ntoks_kept) == n_keep, (
        f"length bookkeeping: {len(ntoks_kept)} lengths for {n_keep} kept rows")

    bounds = _shard_bounds(n_keep, args.gpus)
    print(f"{n_keep} rows -> {args.gpus} GPU shards {bounds[0]}..{bounds[-1]}", flush=True)

    t0 = time.time()
    mp.spawn(_run_shard, args=(args, bounds), nprocs=args.gpus, join=True)
    print(f"extraction done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Merge shard-by-shard with a streaming writer: the full column is
    # n_rows x d_model float32 (~8 GB for the RL split) and concatenating it in
    # memory alongside the source table is the easy way to OOM here.
    base = src.read().slice(0, n_rows).take(pa.array(keep_idx))
    writer = None
    row = 0
    try:
        for rank, (lo, hi) in enumerate(bounds):
            acts = pq.read_table(Path(args.shard_dir) / f"shard_{rank:03d}.parquet")
            assert acts.num_rows == hi - lo, (
                f"shard {rank} has {acts.num_rows} rows, expected {hi-lo}"
            )
            chunk = base.slice(lo, hi - lo).append_column(
                ACTIVATION_COLUMN, acts.column(ACTIVATION_COLUMN)
            )
            if args.layer is not None:
                # Keep the column honest about which layer these vectors are
                # from — train_sft reads it back to check the AR critic depth.
                chunk = chunk.set_column(
                    chunk.schema.get_field_index(LAYER_COL), LAYER_COL,
                    pa.array(np.full(chunk.num_rows, args.layer, dtype=np.int64)))
            if args.cross_model:
                # Same reasoning for the token count: the source model's
                # n_raw_tokens does not describe this parquet any more, and a
                # stale value here would put any later same-model regen (or any
                # consumer that trusts it to locate the final token) on the
                # wrong position.
                chunk = chunk.set_column(
                    chunk.schema.get_field_index(NTOK_COL), NTOK_COL,
                    pa.array(ntoks_kept[lo:hi].astype(np.int64)))
            if writer is None:
                writer = pq.ParquetWriter(args.out, chunk.schema, compression="zstd")
            writer.write_table(chunk)
            row += chunk.num_rows
    finally:
        if writer is not None:
            writer.close()
    assert row == n_keep, f"merged {row} rows, expected {n_keep}"

    # The sidecar is the training-time contract (marker token, templates, scales);
    # it must travel with the regenerated parquet or config load fails. When we
    # retarget a different model/layer it must also be REWRITTEN: the trainers
    # assert d_model and extraction.layer_index against the data (AR derives its
    # critic depth from layer_index+1), so shipping the source model's sidecar
    # would either crash or, worse, silently train a wrong-depth critic.
    s = Path(str(args.inp) + ".nla_meta.yaml")
    if s.exists():
        meta = yaml.safe_load(s.read_text())
        ex = meta.setdefault("extraction", {})
        old = (ex.get("base_model"), ex.get("d_model"), ex.get("layer_index"))
        ex["base_model"] = args.base_model
        ex["d_model"] = d_model
        ex["layer_index"] = layer_index
        if old != (args.base_model, d_model, layer_index):
            meta["dataset_id"] = (
                f"{meta.get('dataset_id', 'nla')}__regen_"
                f"{args.base_model.split('/')[-1]}_L{layer_index}"
            )
            meta["parent_datasets"] = [meta.get("dataset_id")]
            print(f"sidecar retargeted: base_model/d_model/layer_index "
                  f"{old} -> {(args.base_model, d_model, layer_index)}")
        Path(str(args.out) + ".nla_meta.yaml").write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
        print(f"wrote sidecar -> {args.out}.nla_meta.yaml")

    if not args.keep_shards:
        shutil.rmtree(args.shard_dir, ignore_errors=True)
    print(f"wrote {args.out} ({row} rows) in {(time.time()-t0)/60:.1f} min total")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
