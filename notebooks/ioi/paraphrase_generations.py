"""Paraphrase cached AV explanations, so the swap matrix can be re-scored on
reworded text.

The swap matrix asks whether AR_j reads AV_i's explanations as well as its own.
A flat matrix says the arms share a language; it does NOT say what that language
is made of. Paraphrasing every explanation before scoring separates the two
readings: if FVE survives a rewrite that keeps only ~30% of the original words,
the AR is reading CONTENT, not a surface code that happens to be shared.

Method is carried over unchanged from an earlier IOI paraphrase experiment so
the two results are comparable: the same two prompt templates, the same paraphraser
(`google/gemma-3-27b-it`), greedy decode, and the same word-overlap statistic.
There it gave mean overlap 0.372 (medium) / 0.284 (heavy) with IOI accuracy
1.000 at both levels. Expect DIFFERENT overlaps here — those explanations were
one or two sentences and these run 100-120 words — which is exactly why the
measured overlap is recorded per row rather than assumed from the label.

The paraphraser is deliberately NOT one of the arms' own base model. All four
arms share Qwen2.5-7B-Instruct, so using it would be symmetric and defensible
too, but an outside model keeps "the rewriter has no stake in any arm" true by
construction rather than by argument.

Output is drop-in for the scorer. Each level gets its own out-dir laid out the
way `eval_matrix.py` expects, so scoring needs no new code path:

    python notebooks/ioi/paraphrase_generations.py \
        --gen-dir notebooks/ioi/results/paraphrase/generations/temp0 \
        --out-root notebooks/ioi/results/paraphrase/generations \
        --tag temp0

    # -> notebooks/ioi/results/paraphrase/generations/temp0_para_medium/generations/nla1.json ...
    python notebooks/ioi/eval_matrix.py --phase score \
        --out-dir notebooks/ioi/results/paraphrase/generations/temp0_para_medium \
        --nla name=nla1,ar_ckpt=... [...] --rl-parquet ...

`explanation` holds the paraphrase (that is the field the scorer reads);
`explanation_orig` and `overlap` ride along so nothing is lost and the intensity
is auditable after the fact.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PARAPHRASER = "google/gemma-3-27b-it"

# Verbatim from the earlier IOI paraphrase experiment. Do not "improve" these: the whole
# point of reusing them is that the resulting overlap levels are comparable to
# the IOI numbers. A reworded prompt is a different condition.
PARAPHRASE_PROMPT_MEDIUM = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)
PARAPHRASE_PROMPT_HEAVY = (
    "Please paraphrase aggressively. Preserve meaning and uncertainty, but "
    "substantially change wording, syntax, and organization. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "Text:\n{text}\n\n"
    "Respond with only the paraphrased text, wrapped exactly like this:\n"
    "<transformed_text>your paraphrase here</transformed_text>"
)
# The fidelity clause. Measured on the two levels above, gemma-3-27b-it drops 23%
# of numbers and 34% of proper nouns at medium (37% / 45% at heavy) — it is
# complying with "substantially different words and sentence structure", which
# says nothing about keeping specifics. A critic reconstructing an activation over
# one document cannot be handed a text with that document's specifics removed, so
# FVE falls whether or not surface form mattered.
#
# Appended to BOTH prompts above rather than replacing them: their wording is what
# makes the overlap levels comparable to the IOI result, and keeping the
# unfaithful pair is what lets the write-up show why the naive version of this
# experiment overstates the effect. Intensity x fidelity, four cells.
FIDELITY_CLAUSE = (
    "\n\nPreserve every specific detail. Do not remove, alter, generalise or "
    "summarise any fact, number, date, name, or quoted string -- every specific "
    "that appears in the original must appear in your paraphrase. Rewrite how it "
    "is said, not what is said."
)


def _with_fidelity(template: str) -> str:
    """Insert the clause after the instruction, before the text being rewritten —
    an instruction placed after the payload is easy for the model to skip."""
    for marker in ("<text>", "Text:"):
        if marker in template:
            head, sep, tail = template.partition(marker)
            return head.rstrip() + FIDELITY_CLAUSE + "\n\n" + sep + tail
    raise AssertionError("no payload marker in template")


LEVELS = {"medium": PARAPHRASE_PROMPT_MEDIUM, "heavy": PARAPHRASE_PROMPT_HEAVY,
          "medium_faithful": _with_fidelity(PARAPHRASE_PROMPT_MEDIUM),
          "heavy_faithful": _with_fidelity(PARAPHRASE_PROMPT_HEAVY)}

_TAG = re.compile(r"<transformed_text>(.*?)</transformed_text>", re.DOTALL)


def extract_transformed(raw: str) -> tuple[str, bool]:
    """(text, tag_found). Falls back to the raw completion, as the IOI code does
    — a missing tag is usually the model answering without the wrapper, not a
    refusal, and dropping the row would silently bias the set toward explanations
    the paraphraser found easy."""
    m = _TAG.search(raw)
    return (m.group(1).strip(), True) if m else (raw.strip(), False)


def word_overlap(original: str, paraphrased: str) -> float:
    """Fraction of the original's distinct words that survive, as in the
    earlier IOI paraphrase experiment. Set-based and case-folded, so it measures vocabulary
    reuse rather than order — a genuine restructuring scores low even if it keeps
    the same content words."""
    orig = set(original.lower().split())
    if not orig:
        return 0.0
    return len(orig & set(paraphrased.lower().split())) / len(orig)


def load_paraphraser(model_id: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
        )
    except ValueError:
        # Gemma-3 ships a multimodal config; on some transformers versions the
        # causal-LM auto class refuses it and the image-text class is the one
        # that resolves. The text tower is identical either way.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa",
        )
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts: list[str], max_new_tokens: int) -> list[str]:
    """padding_side MUST be left for a decoder-only model: right padding puts pad
    tokens between the prompt and the first generated token and corrupts the
    continuation. The tokenizer default is right, so flip and restore."""
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        tok.padding_side = prev_side
    n_in = enc["input_ids"].shape[1]
    return [tok.decode(row[n_in:], skip_special_tokens=True).strip() for row in out]


def paraphrase_rows(tok, model, rows: list[dict], level: str, args) -> list[dict]:
    """One pass over an arm's explanations at one intensity level.

    Rows with no explanation are passed through unchanged. Exactly one exists —
    AV=nla3, greedy, idx 94, whose generation hit the token cap and never closed
    its tag, so `extract_explanation` returned None. Paraphrasing it would rewrite
    the literal string "None"; the scorer must keep seeing it as unextractable
    (reward -2.0, excluded from the mean) so the paraphrased matrix rests on the
    same rows as the original one. Marked `paraphrase_skipped` so the pass-through
    is visible in the output rather than inferred from a null.
    """
    template = LEVELS[level]
    out_rows: list[dict] = [dict(r) for r in rows]
    for r in out_rows:
        r["paraphrase_level"] = level
    todo = [i for i, r in enumerate(rows) if (r.get("explanation") or "").strip()]
    for i in set(range(len(rows))) - set(todo):
        out_rows[i]["paraphrase_skipped"] = True
        print(f"      [skip] idx={rows[i].get('idx')} has no explanation", flush=True)
    t0 = time.time()
    for start in range(0, len(todo), args.batch_size):
        idxs = todo[start:start + args.batch_size]
        prompts = [template.format(text=rows[i]["explanation"]) for i in idxs]
        completions = generate_batch(tok, model, prompts, args.max_new_tokens)
        for i, raw in zip(idxs, completions):
            para, tagged = extract_transformed(raw)
            new = out_rows[i]
            new["explanation"] = para              # the field the scorer reads
            new["explanation_orig"] = rows[i]["explanation"]
            new["overlap"] = word_overlap(rows[i]["explanation"], para)
            if not tagged:
                new["tag_missing"] = True
        done = start + len(idxs)
        print(f"      {done}/{len(todo)} rows | {time.time() - t0:6.1f}s", flush=True)
    return out_rows


def summarise(rows: list[dict], threshold: float) -> str:
    done = [r for r in rows if not r.get("paraphrase_skipped")]
    ovs = [r["overlap"] for r in done]
    mean_ov = sum(ovs) / len(ovs) if ovs else float("nan")
    high = sum(1 for o in ovs if o >= threshold)
    missing = sum(1 for r in done if r.get("tag_missing"))
    empty = sum(1 for r in done if not (r["explanation"] or "").strip())
    skipped = len(rows) - len(done)
    return (f"mean overlap {mean_ov:.3f} | >={threshold} (under-paraphrased): "
            f"{high}/{len(done)} | no tag: {missing} | empty: {empty} | "
            f"passed through: {skipped}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-dir", required=True, type=Path,
                   help="Directory of cached generation sets (nla1.json ...), "
                        "e.g. notebooks/ioi/results/paraphrase/generations/temp0")
    p.add_argument("--out-root", required=True, type=Path,
                   help="Parent for the per-level output dirs")
    p.add_argument("--tag", default=None,
                   help="Prefix for output dirs (default: --gen-dir's name). "
                        "Writes <out-root>/<tag>_para_<level>/generations/")
    p.add_argument("--levels", default="medium,heavy",
                   help="Comma-separated subset of: " + ",".join(LEVELS))
    p.add_argument("--model", default=PARAPHRASER)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Explanations per generate call. 8 is conservative for "
                        "27B on one 80GB card (~54GB weights); raise if headroom allows.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="These explanations run 100-120 words, so 512 leaves room "
                        "for a longer paraphrase plus the tag wrapper.")
    p.add_argument("--overlap-threshold", type=float, default=0.7,
                   help="Rows at or above this kept too much vocabulary; reported, "
                        "not dropped (ioi_intervention.py uses the same 0.7).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N rows per arm — smoke test.")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--overwrite", action="store_true",
                   help="Redo levels whose output file already exists (default: skip, "
                        "so an interrupted run resumes at arm/level granularity).")
    return p.parse_args()


def main():
    args = parse_args()
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    assert all(lv in LEVELS for lv in levels), f"unknown level in {levels}"
    tag = args.tag or args.gen_dir.name

    gen_files = sorted(args.gen_dir.glob("*.json"))
    assert gen_files, f"no generation sets in {args.gen_dir}"

    # Work out what is actually left to do BEFORE loading 54GB of weights.
    todo = []
    for level in levels:
        out_dir = args.out_root / f"{tag}_para_{level}" / "generations"
        for src in gen_files:
            dst = out_dir / src.name
            if dst.exists() and not args.overwrite:
                print(f"[skip] {dst} exists", flush=True)
                continue
            todo.append((level, src, dst))
    if not todo:
        print("[done] nothing to do (use --overwrite to redo)", flush=True)
        return

    print(f"[load] {args.model} ({args.dtype})", flush=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tok, model = load_paraphraser(args.model, dtype)

    for level, src, dst in todo:
        rows = json.loads(src.read_text())
        if args.limit:
            rows = rows[:args.limit]
        print(f"[para:{level}] {src.name} — {len(rows)} rows", flush=True)
        out_rows = paraphrase_rows(tok, model, rows, level, args)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temp file: an interrupted write would otherwise leave a
        # truncated JSON that the resume check above treats as complete.
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out_rows, indent=1))
        tmp.replace(dst)
        print(f"[para:{level}] {src.name} -> {dst}", flush=True)
        print(f"      {summarise(out_rows, args.overlap_threshold)}", flush=True)

    print("\n=== summary (mean word overlap with the original) ===", flush=True)
    for level in levels:
        out_dir = args.out_root / f"{tag}_para_{level}" / "generations"
        for f in sorted(out_dir.glob("*.json")):
            rows = json.loads(f.read_text())
            print(f"  {level:6s} {f.stem:6s}  {summarise(rows, args.overlap_threshold)}")


if __name__ == "__main__":
    main()
