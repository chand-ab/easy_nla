"""Edit the AV's explanation text, re-encode it, and patch: does the name in the words drive the answer?

`ioi_intervention.py` measures whether AR(AV(h)) preserves the IOI answer. Where
it does, the explanation almost always contains the answer name. That has two
readings: (a) the name in the text is what carries the answer through the round
trip, or (b) explanations that name the answer are simply the ones whose
activations reconstruct well. This script separates them by editing the text
while holding the prompt and the activation fixed. No verbalizer runs: the
explanations are read from `ioi_intervention.py`'s output, so each condition
costs one AR forward and one short base-model generation (~0.65 s).

Rows are split into two groups by what the unedited round trip did.

**Group A — rows the round trip got RIGHT, whose explanation names the answer.**
Replace every occurrence of the answer name and see whether the answer follows:

    orig                the explanation unmodified
    swap_s              answer -> S, the other name in this prompt
    swap_in1..3         answer -> a name the model can see in this prompt but which
                        plays no role in the final question (a demo name)
    swap_out1..3        answer -> a name that appears nowhere in this prompt
    swap_*_x3           the same swap, the new name written 3 times at every site
                        (the dose arm; `--dose` sets the copy counts)
    replace_s           the whole explanation replaced by one sentence naming S
    replace_in1..3      ... naming the same in-prompt names as swap_in1..3
    replace_out1..3     ... naming the same absent names as swap_out1..3

Rows whose explanation never names the answer are skipped: every group-A edit
on such a row is a no-op, and the figure does not draw them.

**Group B — rows it got WRONG.** Add the missing answer name and see whether the
answer is rescued:

    append_gold / append_s / append_none    one sentence at the end
    append2_gold                            the same sentence twice
    prepend_gold                            at the start instead
    replace_gold / replace_s                the sentence INSTEAD of the explanation
    replace_bare_gold / replace_bare_s      the bare name instead of the explanation

`append_none` is the sentence with the name removed, so an effect of `append_gold`
can be told apart from "appending any sentence perturbs the vector".

Every variant is scored two ways: `pred_is_*` reads the first word of a six-token
continuation; `tok1_is_*` reads only the patched token, which is the only token
the patch determines. The figure uses `tok1_is_*`.

Decoding is greedy, and the swap names are drawn from a per-row seed, so a rerun
reproduces the same archive.

Inputs are the two files `ioi_intervention.py` wrote for this (layer, shots)
cell. Output goes to results/ioi_text_intervention_<shots>shot_L<layer>.json.

    python notebooks/ioi/ioi_text_intervention.py \
        --ar-ckpt       <nla-L40>/rl_vllm/critic_latest \
        --sidecar       <nla-L40>/rl_vllm --layer 40 \
        --explanations  notebooks/ioi/results/ioi_verbalizations_4shot_L40.json \
        --archive       notebooks/ioi/results/ioi_patch_conditions_4shot_L40.json \
        --device cuda:0 --ar-device cuda:1

`--k-positions K` runs the same arms over the last K tokens, reading the
`_k<K>.json` files a `--k-positions K` run of `ioi_intervention.py` wrote; the
edit is applied to all K explanations. The published figure uses K = 1.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# `nla` and `scripts` are imported from the repo root, which is two levels up.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from ioi_common import (  # noqa: E402
    IOI_DATA,
    first_token_id,
    first_token_ids,
    load_json_pairs,
    strip_word,
)
# The patch hook is ioi_intervention's, not a second copy of it: the two scripts
# have to write into the residual stream in exactly the same way.
from ioi_intervention import LastTokensPatch  # noqa: E402

# The AV's own idiom for the clause that names the next token — matched so the
# inserted sentence sits in the distribution the AR was trained to read.
SENT = 'Final token "to" is a preposition mid-sentence, immediately expecting "{name}" as the answer.'
SENT_NONAME = 'Final token "to" is a preposition mid-sentence, immediately expecting the answer.'


def name_vocabulary(pairs) -> list[str]:
    """Every name the IOI dataset uses, sorted for determinism."""
    vocab = set()
    pat = re.compile(r"\b([A-Z][a-z]+) and ([A-Z][a-z]+)\b")
    for p in pairs:
        m = pat.search(p["input"])
        if m:
            vocab.update(m.groups())
        vocab.add(p["output"])
    return sorted(vocab)


# Draws per arm. Three was the old arm's replication and there is no reason to
# change it: the four swap names ARE the replication, since decoding is greedy.
N_SWAP = 3


def prompt_names(prompt: str, vocab) -> list[str]:
    """Dataset names that literally occur in this prompt, demo blocks included.

    Word-boundary, not substring: "Ann" must not match inside "Anna", or the
    absent pool would silently lose names it is entitled to draw.
    """
    return [v for v in vocab if re.search(rf"\b{re.escape(v)}\b", prompt)]


def swap_name(text: str, old: str, new: str, repeat: int = 1) -> tuple[str, int]:
    """Replace EVERY occurrence of `old` with `repeat` copies of `new`.

    Returns (text, n_replaced) where n_replaced counts SITES, not copies — so it
    stays comparable to `n_gold_in_expl` across doses.

    All occurrences, not the first: these explanations name the gold four to six
    times, and leaving some behind would test nothing.

    `repeat > 1` is the dose arm. Space-joined, which is the AV's own idiom for a
    repeated name: the rows where an absent name already steers the answer are
    the ones whose verbalization collapsed into "Virginia Virginia Virginia, ...".
    Substituted via a lambda, not the replacement-string form, so a name is never
    reinterpreted as a group reference.
    """
    pat = re.compile(rf"\b{re.escape(old)}\b")
    rep = " ".join([new] * repeat)
    return pat.sub(lambda _m: rep, text), len(pat.findall(text))


def expl_ladder(rec: dict, k: int) -> list[str]:
    """The k explanations for one row, final token FIRST.

    Index j is offset -(j+1). A k=1 verbalizations file has only `explanation`;
    a k>1 one carries `explanations` as well, and the two agree at index 0 by
    construction, which is asserted rather than assumed.
    """
    got = rec.get("explanations") or [rec["explanation"]]
    assert got[0] == rec["explanation"], (
        f"row {rec['row']}: explanations[0] is not the final token's explanation")
    assert len(got) >= k, (
        f"row {rec['row']} has {len(got)} verbalizations, need {k} — "
        f"--explanations must be the file a --k-positions {k} run wrote")
    return got[:k]


def build_variants(rec, group: str, vocab, rng) -> dict:
    """{variant_name: ([edited text per position], injected_name_or_None)}.

    `rec["expls"]` is the ladder from `expl_ladder`: entry 0 is the final
    token's explanation, entry j the one for offset -(j+1). EVERY edit is
    applied to EVERY position. The question the arm asks is whether the name in
    the words steers the answer, and at k>1 "the words" are the model's
    description of all k patched positions — editing only the last of them would
    hold the other k-1 fixed as unedited round-trips, which is a different
    experiment. Note what that costs: the number of name copies now scales with
    k, so a k effect and a dose effect are not separable from these bars alone.
    The dose arm (`--dose`) is what separates them, at fixed k.
    """
    gold, s = rec["gold"], rec["s"]
    expls = rec["expls"]
    out = {"orig": (list(expls), None)}

    def swap_all(new_name):
        return [swap_name(e, gold, new_name)[0] for e in expls]

    if group == "A":
        # The two arms differ ONLY in whether the injected name is somewhere in
        # the prompt. Both exclude gold and S (whose substitution is `swap_s` or
        # a no-op) and any name the explanation already contains (see module
        # docstring). Drawn from one rng in a fixed order, so a rerun at the same
        # seed reproduces both arms.
        #
        # "already contains" is across ALL k explanations, not just the final
        # token's: a name position -2 has written is no cleaner an edit there
        # than at -1, and a pool that differed per position would make the k
        # bars incomparable to each other.
        in_expl = {v for v in vocab
                   if any(re.search(rf"\b{re.escape(v)}\b", e) for e in expls)}
        blocked = in_expl | {gold, s}
        present = [v for v in prompt_names(rec["prompt"], vocab) if v not in blocked]
        absent = [v for v in vocab if v not in blocked and v not in present]
        out["swap_s"] = (swap_all(s), s)
        # 0-shot has no demos: the prompt holds gold and S and nothing else, so
        # `present` is empty and this arm cannot exist there. Take what there is
        # rather than raising — the plot draws an absent bar as absent, not zero.
        for i, nm in enumerate(rng.sample(present, min(N_SWAP, len(present))), 1):
            out[f"swap_in{i}"] = (swap_all(nm), nm)
        for i, nm in enumerate(rng.sample(absent, min(N_SWAP, len(absent))), 1):
            out[f"swap_out{i}"] = (swap_all(nm), nm)
        return out

    for tag, nm in (("gold", gold), ("s", s)):
        sent = SENT.format(name=nm)
        out[f"append_{tag}"] = ([f"{e}\n{sent}" for e in expls], nm)
        out[f"replace_{tag}"] = ([sent] * len(expls), nm)
        # The bare name with no sentence around it. Separates "the AR needs the
        # name" from "the AR needs the name IN the clause that frames it as the
        # next token" — the AR was trained on prose, and a lone proper noun is
        # the most off-distribution input in the whole study.
        out[f"replace_bare_{tag}"] = ([nm] * len(expls), nm)
    out["append_none"] = ([f"{e}\n{SENT_NONAME}" for e in expls], None)
    sg = SENT.format(name=gold)
    out["append2_gold"] = ([f"{e}\n{sg}\n{sg}" for e in expls], gold)
    out["prepend_gold"] = ([f"{sg}\n{e}" for e in expls], gold)
    return out


def dose_variants(rec: dict, doses, names_for_row: dict[str, str]) -> dict:
    """{swap_in1_x3: (text, name)} — the swap arms rewritten at K copies each.

    `names_for_row` is {swap_in1: name, ...} from this row's own `build_variants`
    output, so the ×K bar uses exactly the name of the bar beside it. Everything
    else is held: same row, same prompt, same activation, same substitution
    SITES. Only the number of copies per site moves.
    """
    out = {}
    for arm, nm in sorted(names_for_row.items()):
        for k in doses:
            out[f"{arm}_x{k}"] = (
                [swap_name(e, rec["gold"], nm, repeat=k)[0] for e in rec["expls"]], nm)
    return out


def replace_variants(rec: dict, names_for_row: dict[str, str]) -> dict:
    """Group-A `replace_*`: the explanation thrown away for the naming sentence alone.

    Group B has these too, but only naming gold or S — both of which are IN the
    prompt, so its `replace_gold` cannot separate "the text names the answer"
    from "the text picks a name already in context". These are the same edit
    with the in-prompt / absent split panel A already uses.

    The anchor is `replace_s`, NOT `replace_gold`. Group A is the rows the round
    trip got right, and the unpatched model already answers gold on every one of
    them (166/166 at L40 4-shot) — so a `replace_gold` bar at 1.00 is equally
    consistent with the patch steering perfectly and with it doing nothing, and
    proves neither. S is in the prompt and is NOT the default answer, so a
    `replace_s` success is a patch that actually moved the output. That is what a
    null for `replace_out` has to be read against.

    (Group B has the opposite property, which is why `replace_gold` is a real
    measurement there: its rows are the ones the round trip got WRONG, so gold is
    what the patch has to restore rather than what it gets for free.)

    Names are the swap arms' own, so `replace_out1` and `swap_out1` differ in the
    edit and in nothing else.
    """
    n = len(rec["expls"])
    out = {"replace_s": ([SENT.format(name=rec["s"])] * n, rec["s"])}
    for arm, nm in sorted(names_for_row.items()):
        out[f"replace_{arm[len('swap_'):]}"] = ([SENT.format(name=nm)] * n, nm)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ar-ckpt", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--base-ckpt", default="google/gemma-3-12b-it")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--explanations", type=Path, required=True,
                    help="the verbalizations JSON ioi_intervention.py wrote for this cell")
    ap.add_argument("--archive", type=Path, required=True,
                    help="the patch archive for this cell; its nla_norm outcome defines group A vs B")
    ap.add_argument("--k-positions", type=int, default=1, metavar="K",
                    help="edit and patch the last K positions instead of only the final "
                         "token. Both input files must come from a --k-positions K run")
    ap.add_argument("--ioi-data", type=Path, default=IOI_DATA)
    ap.add_argument("--dose", default="2,3",
                    help="comma-separated copy counts for the swap_*_x<K> arms (default "
                         "'2,3', what the published archives carry; the figure draws "
                         "only x3). '' skips the dose arms")
    ap.add_argument("--limit-a", type=int, default=None,
                    help="cap group A — a smoke-test knob; three rows exercise every code path")
    ap.add_argument("--limit-b", type=int, default=None, help="cap group B")
    ap.add_argument("--max-answer-tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ar-device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    doses = [int(k) for k in args.dose.split(",") if k.strip()]
    assert all(k >= 2 for k in doses), f"--dose values must be >= 2 (1 is the plain swap): {doses}"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nla.config import load_nla_config
    from nla.schema import normalize_activation, resolve_target_scale
    from nla.utils import critic_predict
    from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_model
    from scripts.show_nla_generations import load_ar_critic

    K = args.k_positions
    assert K >= 1, "--k-positions counts tokens; it starts at 1"
    offsets = list(range(1, K + 1))

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tok)
    layer = args.layer if args.layer is not None else cfg.extraction_layer_index
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    ktag = f"_k{K}" if K > 1 else ""

    expls = json.loads(args.explanations.read_text())
    archive = json.loads(args.archive.read_text())
    # Group A/B is "what the round trip did", and at k>1 that is the k-position
    # round trip — a group split taken from the k=1 archive would silently score
    # a different experiment's rows.
    got_k = archive.get("k_positions", 1)
    assert got_k == K, (
        f"--archive is a k={got_k} run but --k-positions is {K}; pass the "
        f"matching …_k{K}.json")
    # Shot count comes from the archive, not a flag: this run inherits whatever
    # prompt setting produced those explanations, and the name must say so.
    out_path = args.out or (
        HERE / "results" / f"ioi_text_intervention_{archive['n_shots']}shot_L{layer}{ktag}.json")
    correct_rows = {r["row"] for r in archive["records"] if r["conditions"]["nla_norm"]["answer"]}
    pairs = load_json_pairs(args.ioi_data)
    vocab = name_vocabulary(pairs)
    for e in expls:
        e["expls"] = expl_ladder(e, K)
    print(f"[cfg] layer={layer} d_model={cfg.d_model} explanations={len(expls)} "
          f"k_positions={K} correct={len(correct_rows)} vocab={len(vocab)} names "
          f"dose={doses or '-'}")

    # No AV: nothing is generated FROM an activation here, only text -> vector.
    raw = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    base = resolve_text_model(raw).to(args.device).eval()
    layers = resolve_decoder_layers(base)
    device = base.get_input_embeddings().weight.device
    ar_device = args.ar_device or args.device
    critic = load_ar_critic(args.ar_ckpt, args.base_ckpt, ar_device)

    def run_base(prompt, patch=None):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        hook = LastTokensPatch(patch, n_capture=K)
        handle = layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = base.generate(**enc, max_new_tokens=args.max_answer_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id,
                                  return_dict_in_generate=True, output_logits=True)
        finally:
            handle.remove()
        assert hook.fired, f"hook on block {layer} never fired"
        text = tok.decode(o.sequences[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, o.logits[0][0].float(), hook.captured

    def encode(text):
        ids = tok.encode(cfg.critic_prompt_template.format(explanation=text),
                         add_special_tokens=False)
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device=ar_device)
            return critic_predict(critic, x, None, mse_scale_f)[0].float().cpu()

    def score(text, logits, gold, s, injected):
        """Outcome plus the logits of every name that could plausibly win.

        `injected` is the name written INTO the text, which is the whole point:
        `*_is_injected` is the measurement, and gold/S are its reference points.

        `pred_is_*` reads the first word of the continuation; `tok1_is_*` reads
        the patched token only. Only the first token is determined by the patch
        (the rest decode from an untouched cache), so the word metric both
        over-credits a patch that wrecked the position and let the model
        re-answer on token 2, and under-credits one that landed the name's first
        token and then misspelled the rest.
        """
        pred = strip_word(text)
        probs = torch.softmax(logits, dim=-1)
        gid, sid = first_token_id(tok, gold), first_token_id(tok, s)
        top1 = int(logits.argmax())
        rec = {
            "pred": pred, "continuation": text,
            "pred_is_gold": pred.lower() == gold.lower(),
            "pred_is_s": pred.lower() == s.lower(),
            "logit_gold": float(logits[gid]), "logit_s": float(logits[sid]),
            "logit_diff": float(logits[gid] - logits[sid]),
            "io_vs_s": bool(logits[gid] > logits[sid]),
            "p_gold": float(probs[gid]), "p_s": float(probs[sid]),
            "top1": tok.decode([top1]),
            "top1_id": top1,
            "tok1_is_gold": top1 in first_token_ids(tok, gold),
            "tok1_is_s": top1 in first_token_ids(tok, s),
        }
        if injected is not None:
            iid = first_token_id(tok, injected)
            rec.update({
                "injected": injected,
                "pred_is_injected": pred.lower() == injected.lower(),
                "logit_injected": float(logits[iid]),
                "p_injected": float(probs[iid]),
                # Does the injected name beat BOTH names actually in the prompt?
                "injected_beats_gold": bool(logits[iid] > logits[gid]),
                "injected_first_ids": list(first_token_ids(tok, injected)),
                "tok1_is_injected": top1 in first_token_ids(tok, injected),
            })
        return rec

    by_row = {e["row"]: e for e in expls}
    group_a = [by_row[r] for r in sorted(correct_rows) if r in by_row]
    group_b = [e for e in expls if e["row"] not in correct_rows]
    if args.limit_a:
        group_a = group_a[:args.limit_a]
    if args.limit_b:
        group_b = group_b[:args.limit_b]
    print(f"[groups] A (round-trip was correct) n={len(group_a)}   "
          f"B (was wrong) n={len(group_b)}")

    records, n_cond, n_skipped_a = [], 0, 0
    for group, rows in (("A", group_a), ("B", group_b)):
        for i, rec in enumerate(rows):
            rng = random.Random(args.seed + rec["row"])   # per-sample, reproducible
            gold, s = rec["gold"], rec["s"]
            # Across ALL k explanations: the edit is applied to all of them, so
            # "is there a gold occurrence to substitute" is a question about the
            # ladder, not about the final token alone.
            gold_by_pos = [len(re.findall(rf"\b{re.escape(gold)}\b", e))
                           for e in rec["expls"]]
            n_gold = sum(gold_by_pos)
            if group == "A" and n_gold == 0:
                # Nothing to substitute: every edit would be a no-op and the
                # figure only draws rows where the AV wrote the answer name.
                n_skipped_a += 1
                continue
            variants = build_variants(rec, group, vocab, rng)
            if group == "A":
                # The dose and replace arms reuse the names the swap arms just
                # drew, so each of them pairs with the swap bar beside it.
                swap_names = {k: nm for k, (_, nm) in variants.items()
                              if k.startswith(("swap_in", "swap_out")) and nm}
                if doses:
                    variants.update(dose_variants(rec, doses, swap_names))
                variants.update(replace_variants(rec, swap_names))
            base_text, base_logits, caps = run_base(rec["prompt"])
            # Each position is rescaled to ITS OWN ‖h‖. Norms differ across
            # positions by a factor of several, so one shared target would turn
            # the k ladder into a magnitude sweep.
            norms = {off: float(caps[off].norm()) for off in offsets}
            hn = norms[1]
            out = {"group": group, "row": rec["row"], "sample_idx": rec["sample_idx"],
                   "gold": gold, "s": s, "prompt": rec["prompt"],
                   "gold_first_ids": list(first_token_ids(tok, gold)),
                   "s_first_ids": list(first_token_ids(tok, s)),
                   # act_norm / n_gold_in_expl are offset-1 scalars: the final token.
                   "act_norm": hn, "n_gold_in_expl": gold_by_pos[0],
                   "k_positions": K,
                   "act_norm_by_pos": [norms[off] for off in offsets],
                   "n_gold_in_expl_by_pos": gold_by_pos,
                   "n_gold_in_expl_total": n_gold,
                   "baseline": score(base_text, base_logits, gold, s, None),
                   "variants": {}}
            for vname, (vtexts, injected) in variants.items():
                # One AR forward per position, each rescaled to that position's
                # ‖h‖: direction only, as at k=1.
                patch = {off: normalize_activation(encode(vtexts[off - 1]), norms[off])
                         for off in offsets}
                text, logits, _ = run_base(rec["prompt"], patch=patch)
                out["variants"][vname] = score(text, logits, gold, s, injected)
                out["variants"][vname]["text_chars"] = len(vtexts[0])
                if K > 1:
                    out["variants"][vname]["text_chars_by_pos"] = [len(t) for t in vtexts]
                n_cond += 1
            records.append(out)
            preds = "  ".join(f"{k}:{v['pred']!r}" for k, v in out["variants"].items())
            print(f"[{group} {i + 1}/{len(rows)}] row {rec['row']} gold={gold} S={s} | {preds}",
                  flush=True)

            # Rewritten after every row, so a killed run leaves a readable
            # partial archive.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "layer": layer, "seed": args.seed, "base_ckpt": args.base_ckpt,
                "k_positions": K,
                "ar_ckpt": str(args.ar_ckpt), "sentence_template": SENT,
                "sentence_noname": SENT_NONAME,
                "dose": doses,
                # n_a counts every round-trip-correct row, including the ones
                # skipped above; the figure prints "n = kept of n_a".
                "n_a": len(group_a), "n_b": len(group_b),
                "n_a_skipped_no_gold_in_expl": n_skipped_a,
                "records": records,
            }, indent=1))

    print(f"\nwrote {out_path}  ({len(records)} samples, {n_cond} patched conditions "
          f"at k={K}; {n_skipped_a} group-A rows skipped because the explanation "
          f"never names the answer)")


if __name__ == "__main__":
    main()
