"""Offline AV/AR swap-matrix eval.

Scores cell (i, j) = AV_i's explanations read by AR_j, as held-out FVE. The RL
trainer only ever pairs AV_i with AR_i, so it produces the 4 DIAGONAL cells and
nothing else; the off-diagonal cells are what distinguish "all runs converged on
one shared explanation language" (matrix ~flat) from "each run coevolved a
private code" (diagonal >> off-diagonal).

Everything that touches a number is imported from the trainer, not reimplemented:
eval-row selection, prompt building, injection, rollout, critic scoring, FVE
baseline. The one thing this script adds is loading a MISMATCHED (AV_i, AR_j)
pair — a code path training never exercises.

Two phases, because generation depends only on AV and scoring only on AR:
  A. generate: for each NLA i, load AV_i into vLLM, write 1 explanation per eval
     prompt, cache to disk.  -> N generation passes, not N^2
  B. score:    for each AR_j, read every cached explanation set and score it.
     -> N^2 cheap critic forwards over text that already exists.

Phase A mirrors the RESUME path exactly (vLLM serves the run's merged av_hf, the
HF actor is base + the run's RL adapter, then sync_actor_to_vllm pushes the
trained policy in). That path is verified: NLA 2's pause/resume at step 250 gave
72.1% -> 71.7%, inside the eval noise floor.

Validate before trusting: --only-diagonal at n=128, temp 1.0 must reproduce each
run's own last in-run eval (NLA 2 = 73.3%). The diagonal is the only place with a
known-good reference; if it does not reproduce, no off-diagonal number is
meaningful.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import extract_explanation, resolve_target_scale
from nla.utils import build_prompt_text
from nla.val_split import is_val_doc, val_doc_permille


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nla", action="append", required=True, metavar="SPEC",
                   help="Repeatable. name=<n>,av_ckpt=<merged av_hf>,"
                        "av_adapter=<rl iter dir>,ar_ckpt=<critic dir>")
    p.add_argument("--base-ckpt", default="Qwen/Qwen2.5-7B-Instruct",
                   help="RAW base the RL LoRA sits on (NOT the merged av_hf)")
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--sidecar", default=None, help="defaults to --rl-parquet")
    p.add_argument("--val-rows", type=int, default=50000,
                   help="must match training so val_permille matches (default 50000)")
    p.add_argument("--eval-n-prompts", type=int, default=128)
    p.add_argument("--eval-temperature", type=float, default=0.0,
                   help="0 = greedy (default). Use 1.0 to compare against in-run evals.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    # vLLM seeds its sampler from the ENGINE seed (SamplingParams carries no
    # seed=), and that seed defaults to 0 — so two runs of this script at
    # temperature>0 replay the SAME draw, not an independent one. Vary this to
    # measure the sampling spread; irrelevant at --eval-temperature 0 (greedy).
    p.add_argument("--vllm-seed", type=int, default=0)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.40)
    p.add_argument("--vllm-max-len", type=int, default=1024)
    p.add_argument("--critic-batch", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--only-diagonal", action="store_true",
                   help="validation mode: score only (i,i)")
    p.add_argument("--reuse-generations", action="store_true",
                   help="skip phase A for NLAs whose explanation cache already exists")
    # vLLM's EngineCore runs in a spawned subprocess that does NOT release its GPU
    # allocation when the parent drops the LLM handle — a second engine in the same
    # process then tries to allocate on top of the first and stalls. So generation
    # runs ONE NLA PER PROCESS (exit is the only reliable teardown), which also lets
    # the four AVs generate concurrently on separate GPUs.
    p.add_argument("--phase", choices=("generate", "score", "both"), default="both",
                   help="generate: one --nla only, writes its cache, exits. "
                        "score: reads all caches, no vLLM.")
    return p.parse_args()


def parse_nla_spec(spec):
    d = dict(kv.split("=", 1) for kv in spec.split(","))
    missing = {"name", "av_ckpt", "av_adapter", "ar_ckpt"} - set(d)
    assert not missing, f"--nla spec missing {missing}: {spec}"
    return d


def load_eval_rows(rl_parquet, n_prompts, val_rows):
    """Byte-for-byte the trainer's selection (the `is_val_doc` loop that fills
    `eval_rows` in train_rl_vllm.main): the first
    n_prompts rows belonging to held-out docs, in FILE order, no RNG. Identical
    inputs -> identical prompts, which is what makes cells comparable."""
    pf = pq.ParquetFile(rl_parquet)
    total = pf.metadata.num_rows
    permille = val_doc_permille(val_rows, total)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if len(rows) >= n_prompts:
            break
        cols = ["prompt", "activation_vector", "doc_id"]
        rg = pf.read_row_group(rg_idx, columns=cols)
        prompts = rg.column("prompt").to_pylist()
        acts = np.asarray(
            rg.column("activation_vector").combine_chunks().flatten(),
            dtype=np.float32).reshape(len(prompts), -1)
        dids = rg.column("doc_id").to_pylist()
        for i, d in enumerate(dids):
            if is_val_doc(d, permille):
                rows.append({"prompt": prompts[i], "activation": acts[i], "doc_id": d})
                if len(rows) >= n_prompts:
                    break
    print(f"[eval] {len(rows)} held-out-doc prompts (doc-hash split, "
          f"{permille / 10:.1f}% of docs, {len(set(r['doc_id'] for r in rows))} distinct docs)",
          flush=True)
    return rows


def generate_for_nla(nla, rows, cfg_bits, args):
    """Phase A: AV_i writes one explanation per eval prompt.

    Mirrors the trainer's resume path: vLLM holds the run's merged av_hf as the
    weight skeleton, the HF actor carries the RL-trained LoRA on the raw base, and
    sync_actor_to_vllm merges the LoRA into vLLM in place. Serving av_hf alone
    would silently evaluate the SFT policy instead of the RL one.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from vllm import LLM as VLLM

    from nla.train_rl_vllm import rollout_batch_vllm, sync_actor_to_vllm

    inj_id, left_id, right_id, inject_char, tokenizer = cfg_bits
    device = "cuda"

    print(f"[gen:{nla['name']}] base {args.base_ckpt} + adapter {nla['av_adapter']}",
          flush=True)
    actor = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    actor = PeftModel.from_pretrained(
        actor, nla["av_adapter"], adapter_name="default", is_trainable=False)
    actor.set_adapter("default")
    lora_norm = sum(p.detach().float().pow(2).sum().item()
                    for n, p in actor.named_parameters()
                    if "lora_" in n and ".default." in n)
    # A zero norm means the adapter silently failed to load (wrong dir, wrong key
    # names) and we would be scoring the bare SFT policy under the RL run's name.
    assert lora_norm > 0, f"{nla['name']}: RL adapter loaded as all-zero (norm={lora_norm})"
    print(f"[gen:{nla['name']}] sum(default lora_param^2) = {lora_norm:.2e}", flush=True)

    llm = VLLM(
        model=nla["av_ckpt"], tokenizer=nla["av_ckpt"], dtype="bfloat16",
        gpu_memory_utilization=args.vllm_gpu_mem, max_model_len=args.vllm_max_len,
        tensor_parallel_size=1, enforce_eager=True, disable_log_stats=True,
        enable_prefix_caching=False,   # prompts differ only by injected activation
        seed=args.vllm_seed,
    )
    sync_s = sync_actor_to_vllm(actor, llm, ipc=False)
    print(f"[gen:{nla['name']}] RL policy synced into vLLM in {sync_s:.1f}s", flush=True)

    prompts_with_acts = [
        (build_prompt_text(r["prompt"], inject_char, tokenizer),
         torch.tensor(r["activation"], dtype=torch.float32))
        for r in rows
    ]
    t0 = time.time()
    responses = rollout_batch_vllm(
        llm, tokenizer, prompts_with_acts,
        inj_id, 1, args.max_new_tokens, args.eval_temperature,
        left_id=left_id, right_id=right_id,
    )
    by_idx = {r["prompt_idx"]: r["text"] for r in responses}
    out = []
    for i, r in enumerate(rows):
        expl = extract_explanation(by_idx.get(i, ""))
        out.append({"idx": i, "doc_id": r["doc_id"], "explanation": expl,
                    "extracted": expl is not None})
    n_ext = sum(o["extracted"] for o in out)
    print(f"[gen:{nla['name']}] {len(out)} prompts in {time.time() - t0:.1f}s, "
          f"extraction {n_ext}/{len(out)} ({100 * n_ext / len(out):.0f}%)", flush=True)

    del llm, actor
    gc.collect()
    torch.cuda.empty_cache()
    return out


def score_with_ar(ar_ckpt, expl_sets, rows, cfg_bits, baseline, args):
    """Phase B: one AR reads every cached explanation set.

    critic_latest is saved MERGED with clean keys (the save_every block in
    train_rl_vllm.main), so it
    loads as a plain NLACriticModel — no LoRA injection, which is what the RL
    trainer does only because it CONTINUES training the critic.
    """
    from nla.train_rl_vllm import score_with_critic

    _, _, _, _, tokenizer = cfg_bits
    mse_scale_f, template = args._mse_scale_f, args._template
    device = "cuda"

    critic = NLACriticModel.from_pretrained(ar_ckpt, torch_dtype=torch.bfloat16).to(device)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)

    acts = [torch.tensor(r["activation"], dtype=torch.float32) for r in rows]
    results = {}
    for av_name, expls in expl_sets.items():
        rewards = score_with_critic(
            critic, tokenizer, [e["explanation"] for e in expls], acts,
            template, mse_scale_f, device, batch_size=args.critic_batch,
        )
        # Failed extraction / untokenizable -> reward -2.0, EXCLUDED from the mean,
        # exactly as the trainer does (:2306). Reporting n_valid alongside the FVE
        # keeps that exclusion visible instead of silently inflating the score.
        per_row = [(-2.0 if r is None else r) for r in rewards]
        valid = [r for r in per_row if r > -2.0]
        fve = (1.0 - (-float(np.mean(valid))) / baseline) * 100.0 if valid else float("nan")
        results[av_name] = {
            "fve_pct": fve, "n_valid": len(valid), "n_total": len(per_row),
            "reward_mean": float(np.mean(per_row)),
            "per_row": per_row,
        }
        print(f"    AV={av_name:6s} -> FVE {fve:6.2f}%  (valid {len(valid)}/{len(per_row)})",
              flush=True)

    del critic
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    args.sidecar = args.sidecar or args.rl_parquet
    nlas = [parse_nla_spec(s) for s in args.nla]
    out_dir = Path(args.out_dir)
    (out_dir / "generations").mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    # Tokenizer + NLA config from the FIRST av_ckpt, matching the trainer
    # (train_rl_vllm.main loads it from --av-ckpt). All arms share a tokenizer;
    # the sidecar asserts
    # catch family drift.
    tokenizer = AutoTokenizer.from_pretrained(nlas[0]["av_ckpt"])
    cfg = load_nla_config(args.sidecar, tokenizer)
    cfg_bits = (cfg.injection_token_id, cfg.injection_left_neighbor_id,
                cfg.injection_right_neighbor_id, cfg.injection_char, tokenizer)
    args._mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    args._template = cfg.critic_prompt_template
    assert args._template is not None, "critic_prompt_template missing from sidecar"
    print(f"[cfg] inj_id={cfg.injection_token_id} mse_scale_f={args._mse_scale_f} "
          f"d_model={cfg.d_model}", flush=True)

    rows = load_eval_rows(args.rl_parquet, args.eval_n_prompts, args.val_rows)

    # FVE denominator = the eval set's own predict-the-mean variance
    # (`eval_fve_baseline` in train_rl_vllm.main). It depends ONLY on the eval
    # activations, so it is
    # one constant shared by all cells — the matrix is internally comparable.
    from nla.schema import compute_predict_mean_baselines
    act_stack = torch.stack([torch.as_tensor(r["activation"], dtype=torch.float32)
                             for r in rows])
    _, baseline = compute_predict_mean_baselines(act_stack, args._mse_scale_f)
    del act_stack
    print(f"[fve] eval-set baseline = {baseline:.4f}", flush=True)

    # ---- Phase A: generate (one pass per AV, one PROCESS per AV) ----
    if args.phase in ("generate", "both"):
        assert args.phase == "both" or len(nlas) == 1, (
            "--phase generate takes exactly one --nla (one engine per process)")
        for nla in nlas:
            cache = out_dir / "generations" / f"{nla['name']}.json"
            if args.reuse_generations and cache.exists():
                print(f"[gen:{nla['name']}] cache exists, skipping", flush=True)
                continue
            expls = generate_for_nla(nla, rows, cfg_bits, args)
            cache.write_text(json.dumps(expls, indent=1))
        if args.phase == "generate":
            print("[gen] done (phase=generate)", flush=True)
            return

    expl_sets = {}
    for nla in nlas:
        cache = out_dir / "generations" / f"{nla['name']}.json"
        assert cache.exists(), f"missing generation cache for {nla['name']}: {cache}"
        expl_sets[nla["name"]] = json.loads(cache.read_text())

    # ---- Phase B: score (one critic load per AR) ----
    matrix = {}
    for nla in nlas:
        j = nla["name"]
        print(f"[score] AR={j}  ({nla['ar_ckpt']})", flush=True)
        subset = ({j: expl_sets[j]} if args.only_diagonal else expl_sets)
        matrix[j] = score_with_ar(nla["ar_ckpt"], subset, rows, cfg_bits, baseline, args)

    # ---- report ----
    names = [n["name"] for n in nlas]
    summary = {
        "eval_n_prompts": len(rows),
        "distinct_docs": len(set(r["doc_id"] for r in rows)),
        "eval_temperature": args.eval_temperature,
        "fve_baseline": baseline,
        "cells": {f"AV={i}|AR={j}": matrix[j][i]["fve_pct"]
                  for j in names for i in matrix[j]},
    }
    (out_dir / "matrix.json").write_text(json.dumps(
        {"summary": summary,
         "detail": {j: {i: {k: v for k, v in c.items() if k != "per_row"}
                        for i, c in matrix[j].items()} for j in names},
         "per_row": {j: {i: c["per_row"] for i, c in matrix[j].items()} for j in names}},
        indent=1))

    print("\n=== FVE %  (rows = AV writer, cols = AR reader) ===")
    print("        " + "".join(f"{('AR=' + j):>10s}" for j in names))
    for i in names:
        cells = "".join(
            f"{matrix[j][i]['fve_pct']:>10.2f}" if i in matrix[j] else f"{'-':>10s}"
            for j in names)
        print(f"AV={i:5s}" + cells)
    print(f"\nwrote {out_dir / 'matrix.json'}")


if __name__ == "__main__":
    main()
