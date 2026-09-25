"""Round-trip an activation through text and patch it back: does the IOI answer survive?

For each IOI prompt the base model's block-K output at the final token, `h`, is
verbalized by the AV, the explanation is re-encoded by the AR, and the resulting
vector is written back into the residual stream in place of `h`. Seven
conditions are recorded per prompt — the unpatched baseline and six patches:

    baseline    no patch — what the model does on its own
    identity    patch h itself back in — must reproduce baseline exactly
    nla_raw     patch AR(AV(h)) at the AR's own output magnitude
    nla_norm    the same vector rescaled to ‖h‖ — direction only
    mismatch    patch AR(AV(h_j)) from a name-disjoint prompt j, at ‖h‖ — a floor
                that is still a real reconstruction of a real activation
    random      patch a Gaussian vector at ‖h‖ — the floor with no structure
    zero        patch the zero vector — the position deleted

`nla_raw` and `nla_norm` are both run because the AR's output magnitude is never
supervised (the training loss normalizes both sides): at a mid-stack layer it
comes out ~1e-3 × ‖h‖, so rescaling is not cosmetic. At the last block the two
must agree — the only consumer of the patched position is the final RMSNorm,
which is scale-invariant — and that agreement is a check that the patch lands
where this docstring says it does.

Three metrics are recorded for every condition (`plot_ioi_grids.py` picks one):

    answer       first word of the continuation is the IO name
    answer_tok1  the patched token is the first token of the IO name — the
                 default in the figure, since only that token is determined by
                 the patch; everything after it decodes from an untouched cache
    io_vs_s      logit(IO) > logit(S) at the patched position (chance 0.5)
    top1_match   the next token equals the unpatched next token

Outputs, per (layer, shot count), all under results/:

    ioi_patch_conditions_<shots>shot_L<layer>.json   the archive the figure reads
    ioi_verbalizations_<shots>shot_L<layer>.json     the AV's explanations —
                                                     the input to ioi_text_intervention.py
    ioi_vectors_<shots>shot_L<layer>.npz             h and AR(AV(h)) per row

Memory: one 12B base with the AV LoRA on it (the base generations run under
`disable_adapter()`), plus the AR critic as a second model. Measured peak on an
A100 is 47.9 GiB, so use an 80 GB card or split with `--ar-device`.

    python notebooks/ioi/ioi_intervention.py \
        --av-lora  <nla-L40>/rl_vllm/iter_000400 \
        --ar-ckpt  <nla-L40>/rl_vllm/critic_latest \
        --sidecar  <nla-L40>/rl_vllm --layer 40 \
        --n-shots 4 --device cuda:0 --ar-device cuda:1

`--k-positions K` extends the patch to the last K tokens (each verbalized and
reconstructed on its own) and writes one archive per k in 1..K. The published
figures use the default, K = 1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

import numpy as np

# `nla` and `scripts` are imported from the repo root, which is two levels up.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from ioi_common import (  # noqa: E402
    IOI_DATA,
    first_token_ids,
    build_target_prompt,
    disjoint_partner,
    first_token_id,
    io_and_s,
    load_json_pairs,
    strip_word,
)

# (key, axis label). Order is the plotting order.
CONDITIONS = [
    ("baseline", "Baseline\n(no NLA)"),
    ("identity", "Identity patch\nre-inject $h$"),
    ("nla_raw", "NLA round-trip\nAR's own norm"),
    ("nla_norm", "NLA round-trip\nrescaled to $\\|h\\|$"),
    ("mismatch", "Mismatch donor\nAR(AV($h_j$)), $\\|h\\|$"),
    ("random", "Random vector\nof norm $\\|h\\|$"),
    ("zero", "Zero vector\n(position deleted)"),
]
# Scaling applied to each patch, recorded in the archive so a reader never has
# to infer it from the bar labels.
PATCH_SCALING = {
    "identity": "exact (h itself)",
    "nla_raw": "none — the AR's own output magnitude",
    "nla_norm": "rescaled to |h|",
    "mismatch": "rescaled to |h|",
    "random": "rescaled to |h|",
    "zero": "n/a — the zero vector",
}
PATCHED = [k for k, _ in CONDITIONS if k != "baseline"]

# Measured on every run, omitted from the figure. `identity` is an instrument
# check, not a result: it re-injects h unchanged and must reproduce baseline
# exactly. Dropping it from CONDITIONS instead would stop it being computed.
PLOT_SKIP = {"identity"}

METRICS = {
    "answer": ("Accuracy — first word is the IO name", None),
    "answer_tok1": ("Accuracy — the patched token STARTS the IO name", None),
    "io_vs_s": ("P(logit IO > logit S) at the patched token", 0.5),
    "top1_match": ("Next token unchanged from unpatched run", None),
}


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
class LastTokensPatch:
    """Capture, and optionally replace, block-K's output at the last few tokens.

    Positions are addressed by 1-BASED OFFSET FROM THE END: 1 is the final
    token, 2 the one before it. That is the indexing every k-position artifact
    in this study uses, because it is the only one invariant to prompt length —
    an absolute index would mean a different token in every row and in every
    shot count.

    `patch` is either a single tensor (the k=1 case, applied at offset 1) or a
    {offset: tensor} mapping; positions it does not name are left alone.
    `captured` is always the dict {offset: h} for offsets 1..n_capture, so a
    caller reads position -1 as `captured[1]` whatever k it asked for.

    Fires on the PREFILL only (`seq_len > 1`); the single-token decode steps
    that follow are left alone, so exactly the named positions are patched and
    the continuation after them is the model's own.
    """

    def __init__(self, patch=None, n_capture=1):
        # A bare tensor is the k=1 spelling: one vector at the final position.
        self.patch = {} if patch is None else (
            patch if isinstance(patch, dict) else {1: patch})
        self.n_capture = n_capture
        self.captured = {}
        self.fired = False

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.fired or h.shape[1] < 2:
            return output
        self.fired = True
        need = max(self.n_capture, *self.patch) if self.patch else self.n_capture
        assert h.shape[1] > need, (
            f"prompt is {h.shape[1]} tokens, too short to address offset {need}")
        for off in range(1, self.n_capture + 1):
            self.captured[off] = h[0, -off, :].detach().float().clone()
        if not self.patch:
            return output
        h = h.clone()
        for off, v in self.patch.items():
            h[0, -off, :] = v.to(device=h.device, dtype=h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h


def make_base_runner(tok, model, layers, layer, max_answer_tokens, *, disable_adapter,
                     n_capture=1):
    """Greedy continuation of a prompt on the RAW base, patch optional.

    `disable_adapter` is needed here because the base carries the AV LoRA and
    must switch it off for these generations; a caller with a raw base passes
    False.

    Returns (continuation text, logits of the first new token, {offset: h}).
    The third element is a dict keyed by 1-based offset from the end even at
    n_capture=1, where it is just `{1: h}`.
    `output_logits` is the unprocessed distribution — `scores` would be the
    post-processor one, which is the same here only by accident of there being no
    processors configured.
    """
    import torch

    device = model.get_input_embeddings().weight.device
    ctx = model.disable_adapter if disable_adapter else contextlib.nullcontext

    def run_base(prompt: str, patch=None):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        hook = LastTokensPatch(patch, n_capture=n_capture)
        handle = layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad(), ctx():
                out = model.generate(
                    **enc, max_new_tokens=max_answer_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    return_dict_in_generate=True, output_logits=True)
        finally:
            handle.remove()
        assert hook.fired, f"patch/capture hook on block {layer} never fired"
        text = tok.decode(out.sequences[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, out.logits[0][0].float(), hook.captured

    return run_base


def make_scorer(tok):
    """The per-condition record: what the model said and how the two names scored."""
    import torch

    def score(text, logits, io_id, s_id, io_name, base_top1):
        pred = strip_word(text)
        probs = torch.softmax(logits, dim=-1)
        top1 = int(logits.argmax())
        return {
            "continuation": text,
            "pred": pred,
            "answer": pred.lower() == io_name.lower(),
            "logit_diff": float(logits[io_id] - logits[s_id]),
            "io_vs_s": bool(logits[io_id] > logits[s_id]),
            "p_io": float(probs[io_id]),
            "p_s": float(probs[s_id]),
            "answer_tok1": top1 in first_token_ids(tok, io_name),
            "top1_id": top1,
            "top1": tok.decode([top1]),
            "top1_match": base_top1 is None or top1 == base_top1,
        }

    return score


def run(args) -> dict:
    """Run the sweep and return the archive dict (also written to --json)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nla.config import load_nla_config
    from nla.schema import (
        EXPLANATION_RE,
        compute_predict_mean_baselines,
        normalize_activation,
        resolve_target_scale,
    )
    from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
    from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_model
    # The AR checkpoint has two on-disk formats and show_nla_generations already
    # knows both; importing it keeps that knowledge in one place.
    from scripts.show_nla_generations import load_ar_critic

    pairs = load_json_pairs(args.ioi_data)
    rng = random.Random(args.seed)
    # Seeded so a rerun describes the same rows in the same order.
    query_idxs = random.Random(args.seed).sample(range(len(pairs)), min(args.n, len(pairs)))

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tok)
    layer = args.layer if args.layer is not None else cfg.extraction_layer_index
    assert layer is not None, "no --layer and no extraction.layer_index in the sidecar"
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    args.json = args.json or _default_json(layer, args.n_shots)
    assert cfg.critic_prompt_template, "sidecar has no critic prompt template — cannot query the AR"
    print(f"[cfg] base={args.base_ckpt} layer={layer} d_model={cfg.d_model} "
          f"mse_scale={mse_scale_f} marker={cfg.injection_char!r} (id {cfg.injection_token_id})")

    raw = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    # Resolve BEFORE the adapter load: on Gemma-3 the LM is nested under
    # model.language_model.*, so an adapter keyed on model.layers.* matches
    # nothing and PEFT silently random-inits.
    base = resolve_text_model(raw).to(args.device).eval()
    layers = resolve_decoder_layers(base)
    assert 0 <= layer < len(layers), f"layer {layer} out of range ({len(layers)} blocks)"
    device = base.get_input_embeddings().weight.device

    actor = PeftModel.from_pretrained(base, args.av_lora).eval()
    vref: list = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)

    ar_device = args.ar_device or args.device
    critic = load_ar_critic(args.ar_ckpt, args.base_ckpt, ar_device)

    # The AV prompt is row-invariant (datagen writes one constant `prompt`
    # column), so rebuild it from the sidecar template rather than reading a
    # parquet row — these activations come from IOI, not the corpus.
    av_msgs = [{"role": "user",
                "content": cfg.actor_prompt_template.format(injection_char="<INJECT>")}]
    av_ids = torch.tensor([tok.encode(build_prompt_text(av_msgs, cfg.injection_char, tok),
                                      add_special_tokens=False)],
                          dtype=torch.long, device=device)

    K = args.k_positions
    patched = PATCHED
    run_base = make_base_runner(tok, actor, layers, layer, args.max_answer_tokens,
                                disable_adapter=True, n_capture=K)
    score = make_scorer(tok)

    def verbalize(h):
        """(explanation | None, diagnostics).

        A sample with no parseable <explanation> is dropped, and a drop is not
        self-explanatory: an AV that ran past the token cap mid-tag is a knob
        (raise --max-new-tokens), an AV that stopped early having never opened
        the tag is a property of the checkpoint. Recording which one it was
        costs nothing here and is unrecoverable afterwards.
        """
        vref[0] = h.to(device).unsqueeze(0)
        try:
            with torch.no_grad():
                gen = actor.generate(
                    input_ids=av_ids, attention_mask=torch.ones_like(av_ids),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id, return_dict_in_generate=True)
        finally:
            vref[0] = None
        new_ids = gen.sequences[0, av_ids.shape[1]:]
        resp = tok.decode(new_ids, skip_special_tokens=True)
        m = EXPLANATION_RE.search(resp)
        diag = {"n_new_tokens": int(new_ids.shape[0]),
                "hit_token_cap": int(new_ids.shape[0]) >= args.max_new_tokens,
                "opened_tag": "<explanation>" in resp,
                "tail": resp[-160:]}
        return (m.group(1).strip() if m else None), diag

    def reconstruct(expl: str):
        """AR(explanation) -> raw d_model vector, fp32 on CPU.

        Raw, i.e. unnormalized: the AR's output norm is never supervised (the
        loss normalizes both sides), so the magnitude is meaningless and the
        caller rescales.
        """
        ids = tok.encode(cfg.critic_prompt_template.format(explanation=expl),
                         add_special_tokens=False)
        x = torch.tensor([ids], dtype=torch.long, device=ar_device)
        with torch.no_grad():
            return critic_predict(critic, x, None, mse_scale_f)[0].float().cpu()

    # --- pass 1: baseline behaviour, verbalization, reconstruction ----------
    # Two passes because the mismatch condition needs another sample's h_hat,
    # and a shared row set is only knowable once every explanation has either
    # been extracted or failed to be.
    #
    # With K > 1 every one of the last K positions gets its OWN verbalization and
    # its OWN reconstruction — the NLA is a per-activation object, and there is
    # no such thing as one explanation of k activations. This is the expensive
    # half of the run and it scales linearly in K; pass 2 then costs nothing
    # extra per k, because patching the last j positions only reuses vectors
    # this pass already produced. That nesting is why one --k-positions 4 run
    # yields k = 1, 2, 3 and 4 rather than four separate sweeps.
    offsets = list(range(1, K + 1))
    stage1, dropped = [], []
    for n, qi in enumerate(query_idxs):
        row = pairs[qi]
        io_name, s_name = io_and_s(row)
        prompt = build_target_prompt(pairs, qi, args.n_shots, rng, args.prompt_format)
        text, logits, caps = run_base(prompt)
        io_id, s_id = first_token_id(tok, io_name), first_token_id(tok, s_name)
        base_score = score(text, logits, io_id, s_id, io_name, None)

        # All K or none: a row missing its offset-3 explanation cannot appear in
        # the k=3 archive, and letting it appear in k=1 and k=2 alone would make
        # the four archives disagree about their own row set — which is exactly
        # the comparison the k ladder is for.
        expls, hats_here, diags = {}, {}, {}
        for off in offsets:
            expls[off], diags[off] = verbalize(caps[off])
        missing = [off for off in offsets if expls[off] is None]
        if missing:
            off = missing[0]
            dropped.append({"row": qi, "sample_idx": n, "offset": off, **diags[off]})
            print(f"[{n + 1}/{len(query_idxs)}] row {qi}: no <explanation> at offset -{off} "
                  f"— dropped ({diags[off]['n_new_tokens']} tok, "
                  f"cap={diags[off]['hit_token_cap']}, "
                  f"opened_tag={diags[off]['opened_tag']})", flush=True)
            continue
        for off in offsets:
            hats_here[off] = reconstruct(expls[off])

        def _q(h, h_hat):
            """(cos, mse) between an activation and its reconstruction, normalized."""
            hn = normalize_activation(h.cpu(), mse_scale_f)
            pn = normalize_activation(h_hat, mse_scale_f)
            return (float((pn @ hn) / (pn.norm() * hn.norm())),
                    float(((pn - hn) ** 2).mean()))

        per_pos = {}
        for off in offsets:
            cos, mse = _q(caps[off], hats_here[off])
            per_pos[off] = {
                "offset": off,
                "act_norm": float(caps[off].norm()),
                "recon_norm": float(hats_here[off].norm()),
                "cos": cos, "mse_nrm": mse,
                "explanation": expls[off],
            }
        # The offset-1 quantities stay top-level and unprefixed: offset 1 is
        # what every reader means by "the" activation.
        stage1.append({
            "sample_idx": n, "row": qi, "io": io_name, "s": s_name, "prompt": prompt,
            "io_id": io_id, "s_id": s_id,
            **{k: per_pos[1][k] for k in ("act_norm", "recon_norm", "cos", "mse_nrm",
                                          "explanation")},
            "by_pos": [per_pos[off] for off in offsets],
            "h": {off: caps[off].cpu() for off in offsets},
            "h_hat": hats_here,
            "baseline": base_score,
        })
        print(f"[{n + 1}/{len(query_idxs)}] row {qi} IO={io_name} S={s_name} | "
              f"base pred={base_score['pred']!r} ok={base_score['answer']} "
              f"ld={base_score['logit_diff']:+.2f} | "
              + "  ".join(f"-{off}: cos={per_pos[off]['cos']:.3f} "
                          f"mse={per_pos[off]['mse_nrm']:.4f}" for off in offsets),
              flush=True)

    assert stage1, "every sample failed explanation extraction"
    keep = [r["row"] for r in stage1]
    hats = {r["row"]: r["h_hat"] for r in stage1}

    # Written HERE, not at the end: both are fully determined by pass 1, and a
    # pass-2 crash 90 minutes in should not cost the verbalizations, which are
    # the expensive half of the run (one 512-token generation per sample, per
    # position). `row` is the line number in ioi_1.jsonl and the only safe join
    # key — sample_idx orderings differ between runs. `explanation` is offset 1;
    # `explanations` is the full ladder, present only when K > 1.
    expl_path = args.explanations or _default_out(layer, args.n_shots, "verbalizations",
                                                  ".json", K)
    expl_path.parent.mkdir(parents=True, exist_ok=True)
    expl_path.write_text(json.dumps([
        {"sample_idx": r["sample_idx"], "row": r["row"], "gold": r["io"], "s": r["s"],
         "prompt": r["prompt"], "explanation": r["explanation"],
         **({"k_positions": K,
             "explanations": [p["explanation"] for p in r["by_pos"]]} if K > 1 else {})}
        for r in stage1], indent=1, ensure_ascii=False))
    print(f"wrote {expl_path}  ({len(stage1)} rows x {K} verbalizations)")

    npz_path = args.npz or _default_out(layer, args.n_shots, "vectors", ".npz", K)
    # (n, K, d) when K > 1, (n, d) when K == 1.
    def _stack(field):
        arr = np.stack([np.stack([r[field][off].numpy() for off in offsets])
                        for r in stage1])
        return arr[:, 0, :] if K == 1 else arr

    np.savez_compressed(
        npz_path,
        h_orig=_stack("h"),
        h_hat=_stack("h_hat"),
        offsets=np.array(offsets),
        sample_idx=np.array([r["sample_idx"] for r in stage1]),
        row=np.array([r["row"] for r in stage1]),
    )
    print(f"wrote {npz_path}  ({len(stage1)} x {K} x {cfg.d_model} activations + reconstructions)")

    # --- pass 2: the patched conditions ------------------------------------
    # Offset 1 keeps its own generator, drawn once per sample in the original
    # order, so a --k-positions run reproduces a legacy k=1 archive's `random`
    # bar draw for draw rather than merely in distribution. Offsets >= 2 come
    # from a second stream, which cannot perturb the first.
    gen = torch.Generator().manual_seed(args.seed)
    gen_hi = torch.Generator().manual_seed(args.seed + 977)
    records = {j: [] for j in offsets}
    archives = {}
    for i, r in enumerate(stage1):
        donor = disjoint_partner(pairs, keep, i)
        rand = {}
        for off in offsets:
            rand[off] = torch.randn(cfg.d_model, generator=gen if off == 1 else gen_hi)

        def rescale(v, off):
            """Direction of v at ‖h‖ AT THAT POSITION — not at the last token's.

            Norms differ across positions by a factor of several, so one shared
            target would turn the k ladder into a magnitude sweep.
            """
            return normalize_activation(v, float(r["h"][off].norm()))

        # Per position, then sliced per k: patches[key][off]. Built for every
        # condition and then filtered, not built lazily: these are cheap tensor
        # ops next to a generation, and the alternative is six code paths that
        # can drift.
        patches = {
            "identity": {off: r["h"][off] for off in offsets},
            # Raw vs rescaled is a real distinction only where decoder blocks
            # remain downstream. At the LAST block the sole consumer is the
            # final RMSNorm, which is scale-invariant, so these two are expected
            # to agree exactly — running both is the check, not the result.
            "nla_raw": {off: r["h_hat"][off] for off in offsets},
            "nla_norm": {off: rescale(r["h_hat"][off], off) for off in offsets},
            "mismatch": {off: rescale(hats[donor][off], off) for off in offsets},
            "random": {off: rescale(rand[off], off) for off in offsets},
            # Not rescaled — the point of this one is the missing magnitude. It
            # is what `nla_raw` is suspected of amounting to at 1e-3‖h‖.
            "zero": {off: torch.zeros(cfg.d_model) for off in offsets},
        }
        base_top1 = r["baseline"]["top1_id"]
        for j in offsets:
            conds = {"baseline": r["baseline"]}
            for key in patched:
                sub = {off: patches[key][off] for off in range(1, j + 1)}
                text, logits, _ = run_base(r["prompt"], patch=sub)
                conds[key] = score(text, logits, r["io_id"], r["s_id"], r["io"], base_top1)

            if "identity" in conds and not conds["identity"]["top1_match"]:
                # Not fatal — bf16 clone/rounding could in principle flip a
                # near-tie — but it is the one thing that would invalidate every
                # other bar, so it is reported per occurrence rather than only in
                # the summary.
                print(f"  [warn] identity patch (k={j}) changed the top token on row "
                      f"{r['row']}: {r['baseline']['top1']!r} -> "
                      f"{conds['identity']['top1']!r}", flush=True)

            records[j].append({
                **{k: r[k] for k in ("sample_idx", "row", "io", "s", "prompt", "act_norm",
                                     "recon_norm", "cos", "mse_nrm", "explanation")},
                "by_pos": r["by_pos"][:j],
                "k_positions": j,
                "donor_row": donor,
                "conditions": conds,
            })
            print(f"  ({i + 1}/{len(stage1)}) row {r['row']} k={j} " + "  ".join(
                f"{k}:{conds[k]['pred']!r}{'✓' if conds[k]['answer'] else '✗'}"
                for k, _ in CONDITIONS if k in conds), flush=True)

        # Recomputed per sample so a killed run still leaves a readable archive.
        # The denominator moves as rows arrive; only the final one is the number.
        # FVE is an offset-1 quantity in every archive: it describes how well the
        # AR reconstructs, which is a property of the position's activation and
        # not of how many positions the patch covered.
        _, var_nrm = compute_predict_mean_baselines(
            torch.stack([x["h"][1] for x in stage1[:len(records[1])]]), mse_scale_f)
        for j in offsets:
            archives[j] = summarize(records[j], dropped, cfg, layer, args, mse_scale_f,
                                    var_nrm, j)
            jpath = _k_path(args.json, j, K)
            jpath.parent.mkdir(parents=True, exist_ok=True)
            jpath.write_text(json.dumps(archives[j], indent=1))

    for j in offsets:
        print(f"wrote {_k_path(args.json, j, K)}")
    return archives


def summarize(records, dropped, cfg, layer, args, mse_scale_f, var_nrm,
              k_positions=1) -> dict:
    """Per-condition means + reconstruction quality, as the JSON archive.

    FVE uses the repo's paper definition — the raw-variance predict-the-mean
    baseline (the second return of compute_predict_mean_baselines), matching
    train_rl_self_contained.py — computed ON THESE SAMPLES rather than on the
    training corpus, because IOI sentences are a far narrower distribution than
    the training corpus and the corpus denominator would flatter the number.
    """
    mse = float(np.mean([r["mse_nrm"] for r in records]))
    # var_nrm is 0 for a single row (the sample IS its own mean) and float
    # division would raise rather than produce an obviously-missing number.
    fve = 1.0 - mse / var_nrm if var_nrm > 0 else None
    summary = {}
    for key, _ in CONDITIONS:
        if key not in records[0]["conditions"]:
            continue
        summary[key] = {
            m: float(np.mean([r["conditions"][key][m] for r in records])) for m in METRICS
        }
        summary[key]["logit_diff"] = float(
            np.mean([r["conditions"][key]["logit_diff"] for r in records]))
    return {
        "n": len(records),
        "n_dropped": len(dropped),
        "dropped_rows": dropped,
        # How many positions this archive's patch covered: the last
        # `k_positions` tokens of the prompt, offsets 1..k.
        "k_positions": k_positions,
        "layer": layer,
        "n_shots": args.n_shots,
        "prompt_format": args.prompt_format,
        "base_ckpt": args.base_ckpt,
        "av_lora": str(args.av_lora),
        "ar_ckpt": str(args.ar_ckpt),
        "patch_scaling": PATCH_SCALING,
        "seed": args.seed,
        "d_model": cfg.d_model,
        "mse_scale": mse_scale_f,
        "summary": summary,
        "recon": {"mse_nrm": mse, "fve": fve, "var_nrm": float(var_nrm),
                  "cos": float(np.mean([r["cos"] for r in records])),
                  "norm_ratio": float(np.mean([r["recon_norm"] / r["act_norm"] for r in records]))},
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--av-lora", required=True,
                    help="AV LoRA adapter dir (e.g. <nla>/rl_vllm/iter_000400)")
    ap.add_argument("--ar-ckpt", required=True,
                    help="AR critic dir (e.g. <nla>/rl_vllm/critic_latest)")
    ap.add_argument("--sidecar", required=True,
                    help="dir holding nla_meta.yaml (e.g. <nla>/rl_vllm), or the dataset parquet")
    ap.add_argument("--base-ckpt", default="google/gemma-3-12b-it",
                    help="the RAW base the activations are defined against")
    ap.add_argument("--layer", type=int, default=None,
                    help="decoder block to read/patch (default: sidecar extraction.layer_index)")
    ap.add_argument("--ioi-data", type=Path, default=IOI_DATA)
    ap.add_argument("--n", type=int, default=200, help="how many IOI prompts")
    ap.add_argument("--n-shots", type=int, default=0,
                    help="demonstrations in the Q/A prompt; 0 leaves just the query turn")
    ap.add_argument("--prompt-format", choices=("qa", "bare"), default="qa",
                    help="shape of a ZERO-shot prompt: qa = instruction + 'Q: ...\\nA:' "
                         "wrapper (default); bare = the raw IOI sentence")
    ap.add_argument("--k-positions", type=int, default=1, metavar="K",
                    help="patch the last K tokens instead of only the final one; writes one "
                         "archive per k in 1..K (…_k<k>.json). K=1 keeps the plain filenames")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="AV explanation cap; a sample truncated before its closing tag is dropped")
    ap.add_argument("--max-answer-tokens", type=int, default=6,
                    help="base-model continuation length; >1 so a split name still decodes to one word")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ar-device", default=None,
                    help="put the AR critic on its own GPU (default: --device)")
    ap.add_argument("--explanations", type=Path, default=None,
                    help="verbalizations path (default results/ioi_verbalizations_<shots>shot_L<layer>.json)")
    ap.add_argument("--npz", type=Path, default=None,
                    help="vectors path (default results/ioi_vectors_<shots>shot_L<layer>.npz)")
    ap.add_argument("--json", type=Path, default=None,
                    help="archive path (default results/ioi_patch_conditions_<shots>shot_L<layer>.json)")
    args = ap.parse_args()

    assert args.k_positions >= 1, "--k-positions counts tokens; it starts at 1"
    # --json defaults to a layer-named path, and the layer is only known once
    # the sidecar has been read — so run() resolves it.
    run(args)


def _default_json(layer, n_shots=0) -> Path:
    """Archive path, carrying the shot count as well as the layer so an n-shot
    run cannot overwrite the 0-shot archive."""
    return HERE / "results" / f"ioi_patch_conditions_{n_shots}shot_L{layer}.json"


def _default_out(layer, n_shots, kind: str, ext: str, k_positions: int = 1) -> Path:
    """results/ioi_<kind>_<shots>shot_L<layer>[_k<K>]<ext>. The `_k<K>` tag appears
    only for K > 1, so a default run writes the plain filenames."""
    tag = f"_k{k_positions}" if k_positions > 1 else ""
    return HERE / "results" / f"ioi_{kind}_{n_shots}shot_L{layer}{tag}{ext}"


def _k_path(base: Path, j: int, k_positions: int) -> Path:
    """`…_L47.json` -> `…_L47_k3.json`, but only when the run is a k ladder."""
    if k_positions == 1:
        return base
    return base.with_name(f"{base.stem}_k{j}{base.suffix}")


if __name__ == "__main__":
    main()
