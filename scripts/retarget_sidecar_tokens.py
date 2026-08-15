"""Retarget a dataset sidecar's `tokens` block onto a different base model.

`regenerate_activations.py --cross-model` rebuilds the activation column for a
new base model and rewrites the sidecar's base_model / d_model / layer_index.
It does NOT touch the `tokens` block — and that block is tokenizer-specific:

    injection_char            the marker whose embedding is overwritten
    injection_token_id        its ID in the SOURCE model's vocabulary
    injection_*_neighbor_id   the IDs flanking it in the rendered actor prompt
    critic_suffix_ids         the tail of the critic template

Every one of those is wrong under a different tokenizer, and the failure is not
loud enough to rely on. The marker is the sharp edge: Qwen3-8B's `㈎` is a single
token there but tokenizes to THREE tokens under Gemma-3, so the injection hook
would find no valid single-token marker site and the run would train on prompts
whose activation was never actually injected.

None of this requires touching the parquet. Stage 3 stores the marker as the
literal placeholder `<INJECT>` (schema.INJECT_PLACEHOLDER) and the trainers
substitute `cfg.injection_char` at load time (train_sft.py:225), so the rows are
already model-agnostic — only this metadata has to move.

    python scripts/retarget_sidecar_tokens.py \
        --base-model google/gemma-3-12b-it \
        data/gemma12b_L47/*.nla_meta.yaml

Pass --check to verify without writing (exit 1 on any mismatch).
"""

import argparse
import sys
from pathlib import Path

import yaml

from nla.datagen._common import load_tokenizer
from nla.datagen.injection_tokens import compute_critic_suffix_ids, find_injection_token
from nla.schema import compute_canonical_neighbors


def retarget(path: Path, base_model: str, tokenizer, check: bool) -> bool:
    """Rewrite one sidecar's tokens block. Returns True if it already matched."""
    meta = yaml.safe_load(path.read_text())
    tokens = meta.get("tokens")
    assert tokens, f"{path}: no `tokens` block — is this an NLA sidecar?"

    # Guard against retargeting metadata away from the activations it describes:
    # the tokens block must agree with whoever produced the vectors.
    declared = meta.get("extraction", {}).get("base_model")
    assert declared == base_model, (
        f"{path}: extraction.base_model is {declared!r} but --base-model is "
        f"{base_model!r}. Run regenerate_activations.py --cross-model FIRST so the "
        f"vectors and this metadata describe the same model."
    )

    char, tid = find_injection_token(tokenizer)
    ids = tokenizer(char, add_special_tokens=False)["input_ids"]
    assert ids == [tid], (
        f"{base_model}: marker {char!r} tokenizes to {ids}, not a single token {tid}. "
        f"The injection hook overwrites exactly one position, so a multi-token "
        f"marker cannot work."
    )

    templates = meta.get("prompt_templates") or {}
    assert "actor" in templates, f"{path}: sidecar has no actor template"
    left, right = compute_canonical_neighbors(tokenizer, templates["actor"], char, tid)

    new = {
        "injection_char": char,
        "injection_token_id": tid,
        "injection_left_neighbor_id": left,
        "injection_right_neighbor_id": right,
        # Only stages that actually run the critic carry a suffix; preserve the
        # null so we don't invent a contract this split never had.
        "critic_suffix_ids": (
            compute_critic_suffix_ids(tokenizer, templates["critic"])
            if tokens.get("critic_suffix_ids") is not None and "critic" in templates
            else tokens.get("critic_suffix_ids")
        ),
    }

    if new == tokens:
        print(f"{path.name}: already correct for {base_model}")
        return True

    print(f"{path.name}:")
    for k in new:
        old_v, new_v = tokens.get(k), new[k]
        flag = "" if old_v == new_v else "   <-- changed"
        print(f"    {k:28s} {old_v!r} -> {new_v!r}{flag}")

    if not check:
        meta["tokens"] = new
        path.write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
        print(f"    written")
    return False


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sidecars", nargs="+", help="*.nla_meta.yaml paths")
    p.add_argument("--base-model", required=True)
    p.add_argument("--check", action="store_true",
                   help="verify only; exit 1 if any sidecar would change")
    args = p.parse_args()

    tokenizer = load_tokenizer(args.base_model)
    ok = [retarget(Path(s), args.base_model, tokenizer, args.check)
          for s in args.sidecars]

    if args.check and not all(ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
