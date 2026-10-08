"""Extract base-model activations for each NLA's generated explanations.

Reads the cached greedy explanations (one per arm per held-out document) and runs
them through the RAW base model, capturing every layer. Two poolings are saved:

  mean  — mean over real tokens, the neutral summary of the whole explanation
  last  — the last real token, which is the position the AR actually reads

The encoder is the **base** model, not any arm's AV: the instrument has to be the
same for all four arms, or the measurement is confounded with the thing being
measured. No LoRA, no injection — this is plain text in, hidden states out.

All 29 hidden states are kept (embeddings + 28 blocks) because the forward pass
costs the same either way and it removes the need to guess which layer matters.
Note the layer convention: `layer_index=20` in the NLA config means
`model.layers[20]`'s OUTPUT, which is `hidden_states[21]` here — the arrays in this
file are indexed by hidden_states position, so NLA layer L lives at index L+1.

Output: one .npz with mean[N,29,3584], last[N,29,3584] (fp16) plus arm/doc_id/idx.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-dir", required=True,
                   help="directory of <arm>.json generation caches (greedy)")
    p.add_argument("--arms", nargs="+", default=["nla1", "nla2", "nla3", "nla4"])
    p.add_argument("--base-ckpt", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--out", required=True, help="output .npz")
    return p.parse_args()


def main():
    args = parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model.eval()

    by_arm = {arm: json.loads((Path(args.gen_dir) / f"{arm}.json").read_text())
              for arm in args.arms}

    # ONE shared row set: keep only documents extracted in EVERY arm, the same rule
    # plot_swap_matrix.py applies to the matrices (:163). Dropping failures per-arm
    # instead would leave the arm that failed with fewer rows than the others, and
    # every downstream comparison here is between arms — a pairwise classifier fed
    # 128 rows of one arm against 127 of another is comparing row membership as well
    # as content.
    keep = sorted(set.intersection(*(
        {r["idx"] for r in rows if r["extracted"]} for rows in by_arm.values())))
    total = len(next(iter(by_arm.values())))
    for arm, rows in by_arm.items():
        n_ext = sum(1 for r in rows if r["extracted"])
        print(f"[act] {arm}: {n_ext}/{total} extracted", flush=True)
    print(f"[act] rows kept (extracted in every arm): {len(keep)}/{total}", flush=True)

    keep_set = set(keep)
    texts, meta = [], []
    for arm in args.arms:
        for r in by_arm[arm]:
            if r["idx"] not in keep_set:
                continue
            texts.append(r["explanation"])
            meta.append((arm, r["doc_id"], r["idx"]))
    print(f"[act] {len(texts)} explanations over {len(args.arms)} arms", flush=True)

    n_layers = model.config.num_hidden_layers + 1      # + embedding output
    d_model = model.config.hidden_size
    mean_out = np.zeros((len(texts), n_layers, d_model), dtype=np.float16)
    last_out = np.zeros((len(texts), n_layers, d_model), dtype=np.float16)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(texts), args.batch_size):
        chunk = texts[s:s + args.batch_size]
        enc = [tok.encode(t, add_special_tokens=False)[:args.max_len] for t in chunk]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e)
            att[i, :len(e)] = 1
        ids, att = ids.to(dev), att.to(dev)
        with torch.no_grad():
            hs = model(input_ids=ids, attention_mask=att,
                       output_hidden_states=True).hidden_states
        H = torch.stack(hs, dim=1).float()                       # [B, Lyr, T, D]
        m = att[:, None, :, None].float()
        # Mask BEFORE averaging — padding is not part of the explanation, and
        # including it would make the pooled vector depend on batch composition.
        mean = (H * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)     # [B, Lyr, D]
        idx_last = att.sum(dim=1) - 1                            # last REAL token
        last = H[torch.arange(H.size(0), device=dev), :, idx_last, :]
        mean_out[s:s + len(enc)] = mean.cpu().numpy().astype(np.float16)
        last_out[s:s + len(enc)] = last.cpu().numpy().astype(np.float16)
        if (s // args.batch_size) % 5 == 0:
            print(f"  {s + len(enc)}/{len(texts)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, mean=mean_out, last=last_out,
        arm=np.array([m[0] for m in meta]),
        doc_id=np.array([m[1] for m in meta]),
        idx=np.array([m[2] for m in meta], dtype=np.int32),
        n_layers=n_layers, d_model=d_model, base_ckpt=args.base_ckpt,
    )
    print(f"[act] wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)  "
          f"mean/last each [{len(texts)}, {n_layers}, {d_model}]")


if __name__ == "__main__":
    main()
