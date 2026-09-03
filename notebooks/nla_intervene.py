"""NLA round-trip eval with text edits, targeting present-past.

A port of https://github.com/kayleeisokay/natural_language_autoencoders
(notebooks/intervention.py, notebooks/executor.py) adapted to EasyNLA; lines
marked `# EasyNLA:` are the whole delta.

Usage:
    single-gpu:
    python notebooks/nla_intervene.py --nla-dir ./nla-L32/rl_vllm

    2 gpus:
    python notebooks/nla_intervene.py --nla-dir ./nla-L32/rl_vllm \
        --device cuda:2 --ar-device cuda:3 --n 8 --first-token
"""

import argparse
import json
import random
import re
import string
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# EasyNLA: the sidecar / injection / critic contract. kitft's notebook gets the
# equivalent from nla_inference.py's NLAClient + NLACritic.
from nla.config import load_nla_config
from nla.injection import marker_well_formed
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_model

DATA_DIR = Path(__file__).resolve().parent / "datasets"
TENSE_DATA_PATH = DATA_DIR / "present-past-future.json"

PARAPHRASE_PROMPT = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)


# --- Patch hook: replace the residual stream at the last prompt token ---
class LastTokenPatcher:
    def __init__(self, patch_vec=None):
        self.patch_vec = patch_vec
        self.captured = None
        self._done = False  # only patch once

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Only act on the initial multi-token prefill pass, not each
        # single-token decode step that follows.
        if hidden.shape[1] > 1 and not self._done:
            self.captured = hidden[0, -1, :].detach().clone()
            if self.patch_vec is not None:
                hidden = hidden.clone()
                hidden[0, -1, :] = self.patch_vec.to(hidden.dtype).to(hidden.device)
            self._done = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return output


def mse_cos_from_reconstruction(h_hat: torch.Tensor, h_orig: torch.Tensor, mse_scale: float):
    """Per-example, norm-matched metrics (for reporting alongside FVE, not used *in* FVE)."""
    pred = h_hat.float()
    gold = h_orig.float().cpu()
    pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale
    mse = ((pred_n - gold_n) ** 2).mean().item()
    cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
    return mse, cos


def raw_sq_error(h_hat: torch.Tensor, h_orig: torch.Tensor) -> float:
    """||h_orig - h_hat||^2 on raw, unnormalized activations -- this is what
    the paper's L and FVE are defined over, so it must NOT be norm-rescaled
    the way mse_cos_from_reconstruction's inputs are."""
    # EasyNLA: for the AR arms h_hat already carries ||h_orig|| by construction
    # (ar() normalizes to it), so this reduces to 2||h||^2(1-cos) — the magnitude
    # term is zero by construction, not by skill. Only random_vec/mismatch, whose
    # norms are not set from h_orig, contribute any magnitude error.
    assert h_hat.dim() == 1 and h_orig.dim() == 1, f"{h_hat.shape=} {h_orig.shape=}"
    return ((h_orig.float().cpu() - h_hat.float().cpu()) ** 2).sum().item()


def fve(sq_errors: list, h_orig_list: list) -> float:
    """FVE = 1 - E[||h - AR(z)||^2] / E[||h - h_bar||^2], both expectations
    taken over the same sample. h_bar is estimated as the sample mean of
    h_orig_list -- this must be computed over the full run, not per-example."""
    h_stack = torch.stack([h.float().cpu() for h in h_orig_list])  # [N, d]
    h_bar = h_stack.mean(dim=0)
    variance = ((h_stack - h_bar) ** 2).sum(dim=1).mean().item()
    L = sum(sq_errors) / len(sq_errors)
    return 1 - L / variance


def first_token_ids(tok, word: str) -> tuple[int, ...]:
    """Every first-token id that means "the answer starts HERE".

    Gemma tokenizes ' Jerry' and 'Jerry' as different ids, and which one the
    model emits depends on what precedes it — after 'A:' it is the spaced form,
    at the start of a line the bare one. Both are the name beginning at this
    position, so a first-token metric that accepts only the spaced form scores a
    correct immediate answer as wrong (484 rows across this sweep).
    """
    # EasyNLA caveat: for a word the tokenizer SPLITS, the bare branch is the
    # first fragment — GPT-2 gives 'walked' -> ('walk', 'ed'), so bare 'walk' is
    # the PRESENT form and would score a wrong answer as right. Harmless in this
    # prompt format (after "A:" the model emits the spaced form), but check it
    # before reusing first_token_ids on a template that starts a line.
    spaced = tok.encode(" " + word, add_special_tokens=False)
    bare = tok.encode(word, add_special_tokens=False)
    assert spaced and bare, f"empty tokenization for {word!r}"
    return tuple(sorted({spaced[0], bare[0]}))


def load_tense_map(path: Path = TENSE_DATA_PATH) -> dict:
    """Returns {past_form: (base_form, future_form)}, built from the
    input/past/future JSON. Used both to build ICL pairs and to locate +
    rewrite past-tense words inside the AV's verbalization text.
    """
    with open(path) as f:
        entries = json.load(f)
    tense_map = {}
    for e in entries:
        tense_map[e["past"]] = (e["input"], e["future"])
    return tense_map


def find_editable_positions(text: str, tense_map: dict) -> list[tuple[int, int, str]]:
    """Finds whole-word, case-insensitive matches of known past-tense
    forms in `text`. Returns list of (start, end, past_form_as_matched)
    in order of appearance. Word-boundary regex avoids matching "read"
    inside "already", etc.
    """
    positions = []
    for past_form in tense_map:
        for m in re.finditer(rf"\b{re.escape(past_form)}\b", text, flags=re.IGNORECASE):
            positions.append((m.start(), m.end(), m.group(0)))
    positions.sort(key=lambda x: x[0])
    return positions


def apply_future_edits(text: str, num_edits: int, tense_map: dict, rng: random.Random,
                       relevant_words: set = None, noop: bool = False) -> tuple[str, int]:
    """Rewrites up to `num_edits` past-tense words in `text` to their
    future-tense form. If `relevant_words` is given, only candidate
    positions whose matched word is in that set are eligible -- this
    keeps edits confined to words that actually appeared in this
    prompt's demos/query, instead of any of the ~200 past-tense forms
    in tense_map that might show up incidentally in the verbalizer's
    prose (e.g. 'put', 'read', 'cut' as ordinary words, not as the
    task-relevant transformation being described).
    """
    positions = find_editable_positions(text, tense_map)
    if relevant_words is not None:
        positions = [p for p in positions if p[2].lower() in relevant_words]
    if not positions:
        return text, 0

    chosen = rng.sample(positions, min(num_edits, len(positions)))
    chosen.sort(key=lambda x: x[0], reverse=True)

    edited = text
    applied = 0
    for start, end, matched in chosen:
        past_form_key = matched.lower()
        if past_form_key not in tense_map:
            continue
        base_form, future_form = tense_map[past_form_key]
        replacement = matched if noop else future_form
        edited = edited[:start] + replacement + edited[end:]
        applied += 1

    return edited, applied


def build_icl_prompt(tense_map: dict, query_word: str, n_shots: int, rng: random.Random):
    """Standard present->past ICL prompt. Returns (prompt, past_gold,
    future_gold, shot_words) -- shot_words is now returned so the edit
    step can restrict itself to words that actually appeared in this
    prompt, instead of the full tense_map vocabulary.
    """
    pool = [w for w in tense_map if w != query_word]
    shot_words = rng.sample(pool, min(n_shots, len(pool)))
    lines = [f"Q: {tense_map[w][0]}\nA: {w}" for w in shot_words]  # base -> past
    base_form, future_form = tense_map[query_word]
    lines.append(f"Q: {base_form}\nA:")
    instruction = "Answer with a single word or two words only. No punctuation, no explanation."
    prompt = instruction + "\n\n" + "\n\n".join(lines)
    return prompt, query_word, future_form, shot_words


def extract_answer(text: str) -> str:
    first_line = text.strip().split("\n")[0]
    return first_line.strip(string.punctuation + " ")


def is_correct(pred: str, gold: str, case_sensitive: bool) -> bool:
    if not case_sensitive:
        pred, gold = pred.lower(), gold.lower()
    return pred == gold


def is_future_match(pred: str, future_gold: str) -> bool:
    """Loose match: checks the future gold's content word (after 'will')
    appears in the prediction, since generations may include extra words
    around the core answer (e.g. 'will go' vs 'he will go tomorrow').
    """
    pred_norm = pred.lower().strip()
    future_norm = future_gold.lower().strip()
    return future_norm in pred_norm


def load_ar_critic(ar_dir, device):
    # EasyNLA: no counterpart — NLACritic reads its own nla_meta.yaml, which our
    # sidecar does not satisfy. critic_latest / merged/ar_hf are HF-format
    # (config.json + shards + value_head.safetensors); the iter_* dirs are
    # LoRA-format, load those with scripts/show_nla_generations.py.
    ar_dir = Path(ar_dir)
    assert (ar_dir / "config.json").exists(), (
        f"{ar_dir} has no config.json — not an HF-format critic. Point at "
        f"rl_vllm/critic_latest or merged/ar_hf."
    )
    critic = NLACriticModel.from_pretrained(
        str(ar_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    return critic.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # --nla-dir: the downloaded rl_vllm/ dir — nla_meta.yaml + iter_000400/ + critic_latest/
    # --av-lora / --ar-ckpt default to <nla-dir>/iter_000400 and <nla-dir>/critic_latest
    # --ar-device puts the AR on a second GPU; --widen-pool is their widen_pool branch
    # --first-token adds the tok1_*/d_logit columns (our IOI metric); off by default
    p.add_argument("--nla-dir", required=True)
    p.add_argument("--base-ckpt", default="google/gemma-3-12b-it")
    p.add_argument("--av-lora", default=None)
    p.add_argument("--ar-ckpt", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ar-device", default=None)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--n-shots", type=int, default=4)
    p.add_argument("--n-edits", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tense-data", default=TENSE_DATA_PATH, type=Path)
    p.add_argument("--widen-pool", action="store_true")
    p.add_argument("--first-token", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=384)
    # EasyNLA: their MAX_NEW_TOKENS is 256/512; 16 is enough because
    # extract_answer only reads the first LINE, and 7 arms x --n adds up.
    p.add_argument("--max-answer-tokens", type=int, default=16)
    args = p.parse_args()

    nla_dir = Path(args.nla_dir)
    DEVICE, ar_dev = args.device, args.ar_device or args.device
    rng = random.Random(args.seed)
    tense_map = load_tense_map(args.tense_data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    # EasyNLA: nla_meta.yaml pins the marker ids + both prompt templates, and
    # loading it asserts the live tokenizer still reproduces them.
    cfg = load_nla_config(str(nla_dir), tokenizer)
    assert cfg.injection_scale is None, "these checkpoints inject RAW activations"
    MSE_SCALE = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    LAYER_INDEX = cfg.extraction_layer_index

    # EasyNLA: resolve_text_model BEFORE the adapter load — Gemma-3 nests the LM
    # under model.language_model.*, so an adapter keyed on model.layers.* matches
    # nothing and PEFT silently random-inits.
    base = resolve_text_model(AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"))
    # EasyNLA: the AV is a LoRA in THIS process, not a merged model behind an
    # SGLang server. disable_adapter() gives the untouched base back, so one 12B
    # load serves as both AV and target model.
    model = PeftModel.from_pretrained(base.to(DEVICE).eval(),
                                      str(args.av_lora or nla_dir / "iter_000400")).eval()
    layers = resolve_decoder_layers(model.get_base_model())
    # EasyNLA: injection is ADDITIVE and norm-matched at the OUTPUT OF BLOCK 1
    # (h'_p = h_p + ||h_p||·v/||v||), not a replaced embedding row pre-block-0.
    vref = [None]
    register_karvonen_hook(model, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
                           layer_idx=1)
    critic = load_ar_critic(args.ar_ckpt or nla_dir / "critic_latest", ar_dev)

    def av(h: torch.Tensor) -> str:
        # EasyNLA: in-process marker injection, not client.generate(v, ...).
        ptxt = build_prompt_text(
            [{"role": "user", "content": cfg.actor_prompt_template.format(injection_char="<INJECT>")}],
            cfg.injection_char, tokenizer)
        ids = tokenizer.encode(ptxt, add_special_tokens=False)
        assert marker_well_formed(ids, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                  cfg.injection_right_neighbor_id), "marker/template drift"
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        vref[0] = h[None].float().to(DEVICE)   # raw: the AV sees the magnitude
        try:
            with torch.no_grad():
                out = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        finally:
            vref[0] = None
        m = EXPLANATION_RE.search(tokenizer.decode(out[0, x.shape[1]:], skip_special_tokens=True))
        return m.group(1).strip() if m else None

    def ar(text: str, norm: float) -> torch.Tensor:
        # EasyNLA: NOT "raw, unnormalized -- matches h's scale". This AR is
        # direction-only (its MSE normalizes both sides to sqrt(d_model)), so the
        # caller supplies the magnitude — pass ||h|| to patch on h's scale.
        ids = tokenizer.encode(cfg.critic_prompt_template.format(explanation=text),
                               add_special_tokens=False)
        with torch.no_grad():
            v = critic_predict(critic, torch.tensor([ids], dtype=torch.long, device=ar_dev),
                               None, MSE_SCALE)[0].float().cpu()
        return v if MSE_SCALE is None else normalize_activation(v[None], norm)[0]

    def run_with_patch(prompt: str, patch_vec, max_new_tokens=256):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        patcher = LastTokenPatcher(patch_vec)
        handle = layers[LAYER_INDEX].register_forward_hook(patcher)
        try:
            # EasyNLA: disable_adapter() — `model` carries the AV LoRA.
            # EasyNLA: output_logits so the caller can score the FIRST generated
            # token as well as the decoded string (as our IOI runs do).
            with torch.no_grad(), model.disable_adapter():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id,
                                     return_dict_in_generate=True, output_logits=True)
        finally:
            handle.remove()
        assert patcher._done, f"hook on block {LAYER_INDEX} never fired"
        text = tokenizer.decode(out.sequences[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        return text, patcher.captured, out.logits[0][0].float()

    def paraphrase(text: str, max_new_tokens: int = 512) -> str:
        # EasyNLA: theirs takes a `prompt_template` arg to share one function
        # between MEDIUM and HEAVY; this hardcodes their MEDIUM, and the rewrite
        # runs on the base model under disable_adapter(), via the chat template.
        prompt = PARAPHRASE_PROMPT.format(text=text)
        ptxt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ptxt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        with torch.no_grad(), model.disable_adapter():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"<transformed_text>(.*?)</transformed_text>", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        print(f"[paraphrase] WARNING: no <transformed_text> tags. Raw[:200]={raw[:200]!r}")
        return raw

    ARMS = ["baseline", "nla_roundtrip", "tense_edit", "noop_edit", "paraphrase",
            "random_vec", "mismatch"]
    past_correct = {a: 0 for a in ARMS}
    future_correct = {a: 0 for a in ARMS}
    # EasyNLA: first-token variants of the same two questions, plus the
    # continuous logit gap. `tok1_*` is the naming our IOI archives use.
    tok1_past = {a: 0 for a in ARMS}
    tok1_will = {a: 0 for a in ARMS}
    logit_gap = {a: [] for a in ARMS}
    recon = {a: [] for a in ARMS}
    sq_errors = {a: [] for a in ARMS}
    h_origs = []            # the FVE denominator's population — one h per sample
    edits = []
    queries = rng.sample(list(tense_map), min(args.n, len(tense_map)))

    for i, qw in enumerate(queries):
        prompt, past_gold, future_gold, shot_words = build_icl_prompt(
            tense_map, qw, args.n_shots, rng)
        text_baseline, h_orig, logits_baseline = run_with_patch(prompt, None, args.max_answer_tokens)
        foreign = rng.choice([w for w in tense_map if w != qw])
        _, h_mismatch, _ = run_with_patch(
            build_icl_prompt(tense_map, foreign, args.n_shots, rng)[0], None,
            args.max_answer_tokens)
        explanation = av(h_orig)
        assert explanation, "no <explanation> extracted — check the marker injection"

        if args.widen_pool:
            relevant_words = None
        else:
            relevant_words = {w.lower() for w in shot_words} | {qw.lower()}
        # EasyNLA: same seed for both calls, so noop_edit hits the SAME positions.
        edited, n_applied = apply_future_edits(explanation, args.n_edits, tense_map,
                                               random.Random(i), relevant_words=relevant_words)
        nooped, _ = apply_future_edits(explanation, args.n_edits, tense_map,
                                       random.Random(i), relevant_words=relevant_words, noop=True)
        edits.append(n_applied)

        norm = float(h_orig.norm())
        noise = torch.randn(h_orig.shape, generator=torch.Generator().manual_seed(i),
                            dtype=torch.float32)
        patches = {
            "baseline": None,
            "nla_roundtrip": ar(explanation, norm),
            "tense_edit": ar(edited, norm),
            "noop_edit": ar(nooped, norm),
            "paraphrase": ar(paraphrase(explanation), norm),
            "random_vec": normalize_activation(noise[None], norm)[0],
            "mismatch": h_mismatch.float().cpu(),
        }

        print(f"\n--- sample {i + 1}/{len(queries)}  |  query={qw!r}  future_gold={future_gold!r} ---")
        print(f"  explanation: {explanation}")
        if n_applied:
            print(f"  tense_edit ({n_applied} applied): {edited}")
        # EasyNLA: the two golds' first-token ids. "will walk"/"will jump" share
        # a first token, so tok1_will reads as "did the TENSE flip", decoupled
        # from whether the verb is right — which acc_future still checks.
        past_ids = first_token_ids(tokenizer, past_gold) if args.first_token else ()
        will_ids = first_token_ids(tokenizer, future_gold.split()[0]) if args.first_token else ()
        for arm in ARMS:
            if arm == "baseline":
                text, logits = text_baseline, logits_baseline
            else:
                text, _, logits = run_with_patch(prompt, patches[arm], args.max_answer_tokens)
            pred = extract_answer(text)
            past_correct[arm] += is_correct(pred, past_gold, case_sensitive=False)
            future_correct[arm] += is_future_match(pred, future_gold)
            if args.first_token:
                top1 = int(logits.argmax())
                tok1_past[arm] += top1 in past_ids
                tok1_will[arm] += top1 in will_ids
                logit_gap[arm].append(float(max(logits[i] for i in will_ids)
                                            - max(logits[i] for i in past_ids)))
            if patches[arm] is not None:
                recon[arm].append(mse_cos_from_reconstruction(patches[arm], h_orig, MSE_SCALE))
                sq_errors[arm].append(raw_sq_error(patches[arm], h_orig))
            tok1 = f"  tok1={tokenizer.decode([int(logits.argmax())])!r}" if args.first_token else ""
            print(f"  [{arm:<13}] pred={pred!r}{tok1}")
        h_origs.append(h_orig)

    n = len(queries)
    # acc_* score the first LINE as a string. --first-token adds tok1_*/d_logit,
    # which read the FIRST GENERATED TOKEN instead; d_logit = logit(will) -
    # logit(past form), so it moves before tok1_will flips.
    tok1_hdr = (f"{'tok1_past':>10}{'tok1_will':>10}{'d_logit':>9}"
                if args.first_token else "")
    print(f"\n{'arm':<14}{'acc_past':>9}{'acc_fut':>8}{tok1_hdr}"
          f"{'mse_nrm':>9}{'cos':>7}{'fve':>7}")
    for arm in ARMS:
        if recon[arm]:
            mse = sum(m for m, _ in recon[arm]) / len(recon[arm])
            cos = sum(c for _, c in recon[arm]) / len(recon[arm])
            # h_bar is a sample mean, so the denominator is only meaningful once
            # there are enough samples to estimate it — n=2 is arithmetic, not a
            # measurement. Raise --n before reading this column.
            f = f"{fve(sq_errors[arm], h_origs) * 100:>6.0f}%" if n >= 2 else f"{'-':>7}"
            tail = f"{mse:>9.4f}{cos:>7.3f}{f}"
        else:
            tail = f"{'-':>9}{'-':>7}{'-':>7}"
        if args.first_token:
            gap = sum(logit_gap[arm]) / len(logit_gap[arm])
            tok1_cols = (f"{tok1_past[arm] / n:>10.2f}{tok1_will[arm] / n:>10.2f}"
                         f"{gap:>+9.2f}")
        else:
            tok1_cols = ""
        print(f"{arm:<14}{past_correct[arm] / n:>9.2f}{future_correct[arm] / n:>8.2f}"
              f"{tok1_cols}{tail}")
    print(f"\nn={n}, avg_edits_applied={sum(edits) / n:.1f} of {args.n_edits} requested")


if __name__ == "__main__":
    main()
