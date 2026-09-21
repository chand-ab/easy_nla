# IOI: does a natural-language explanation carry enough of an activation to keep the model's answer?

This folder reproduces two figures from the Gemma-3-12B layer study. Both use
the indirect-object-identification (IOI) task:

> Then, Rebecca and Jessie went to the store. Rebecca gave a peach to ___

The answer is **Jessie** (the indirect object, "IO"). **Rebecca** is the
repeated subject ("S"), the name the model has to *not* say.

**Figure 1 — `ioi_patch_conditions_grid.png`.** Take the activation `h` at the
last token of the prompt, have the AV write an explanation of it, have the AR
turn that explanation back into a vector, and patch the vector into the model in
place of `h`. Does the model still answer Jessie? The figure compares this
round trip against controls (the original `h`, a random vector, a vector from an
unrelated prompt, the zero vector) at four layers and five few-shot settings.

**Figure 2 — `ioi_text_intervention_grid.png`.** Same setup, but now the
*text* of the explanation is edited before the AR reads it back — the answer
name is swapped for another name, or a sentence naming the answer is added or
substituted. Does the model's answer follow the edit? This separates "the name in
the words is what carries the answer" from "explanations that happen to name the
answer are simply the ones that reconstruct well".

## What is here

| file | what it does |
|---|---|
| `ioi_intervention.py` | pass 1: verbalize, reconstruct, patch. Writes the archive Figure 1 reads and the explanations pass 2 needs |
| `ioi_text_intervention.py` | pass 2: edit the explanations, re-encode, patch. Writes the archive Figure 2 reads |
| `ioi_common.py` | shared pieces: the dataset, prompt construction, how names are tokenized |
| `plot_ioi_grids.py` | draws both figures from `results/`. Needs only matplotlib and numpy |
| `run_ioi.sh` | runs pass 1 and pass 2 over every layer and shot count |
| `datasets/ioi_1.jsonl` | the 200 IOI prompts, one `{"input", "output"}` per line |
| `results/` | the archives behind the published figures: 20 cells (4 layers × 5 shot counts) × 3 files |

`results/` holds, per cell:

- `ioi_patch_conditions_<shots>shot_L<layer>.json` — pass 1: per prompt, what
  the model said under each condition, plus reconstruction quality (`recon`)
  and per-condition means (`summary`).
- `ioi_verbalizations_<shots>shot_L<layer>.json` — the AV's explanation of each
  prompt's activation. Pass 2 reads these, so the expensive generation step
  never has to be repeated.
- `ioi_text_intervention_<shots>shot_L<layer>.json` — pass 2: per prompt, what
  the model said after each text edit.

The `.npz` files pass 1 also writes (the raw activations and reconstructions)
are not included; nothing here needs them.

## Redraw the figures

No GPU, no model download.

```bash
pip install matplotlib numpy
python notebooks/ioi/plot_ioi_grids.py            # both figures
python notebooks/ioi/plot_ioi_grids.py --figure text
python notebooks/ioi/plot_ioi_grids.py --metric answer
```

Output goes to `notebooks/ioi/figures/`, which is gitignored.

By default bars are scored at the **patched token only** (`answer_tok1`): did
the first generated token start the target name? The patch determines only that
token; everything after it is generated from an untouched cache, so reading the
whole first word (`--metric answer`) can credit a patch that failed and let the
model recover a token later.

## Rerun the experiment

### Install

```bash
git clone https://github.com/chand-ab/easy_nla.git && cd easy_nla
python -m venv .venv && source .venv/bin/activate
pip install -e .          # torch, transformers, peft, safetensors, pyyaml
hf auth login             # google/gemma-3-12b-it is gated
```

### Download the NLAs

One repo per layer: `achand45/gemma-3-12b-it-nla-L{24,32,40,47}`. Each needs
the RL-final AV LoRA (`iter_000400`, ~2 GB), the matching AR (`critic_latest`,
~20 GB) and the sidecar (`nla_meta.yaml`).

```bash
for L in 24 32 40 47; do
  hf download achand45/gemma-3-12b-it-nla-L$L --local-dir ~/nla/L$L \
    --include 'rl_vllm/iter_000400/adapter*' 'rl_vllm/critic_latest/*' 'rl_vllm/nla_meta.yaml'
done
```

### Run

```bash
NLA_DIR=~/nla bash notebooks/ioi/run_ioi.sh
```

`LAYERS`, `SHOTS`, `DEV` and `AR_DEV` are environment variables; the defaults
run all 20 cells with the base model on `cuda:0` and the AR on `cuda:1`. Base
plus AR peak at 48 GB, so one 80 GB card also works (`AR_DEV=cuda:0`).

Cells are independent, so split `LAYERS` across machines to go faster. A
finished cell leaves a marker in `logs/`, and the script skips those on
restart.

Then draw:

```bash
python notebooks/ioi/plot_ioi_grids.py
```

### Run one cell by hand

```bash
python notebooks/ioi/ioi_intervention.py \
    --av-lora ~/nla/L40/rl_vllm/iter_000400 --ar-ckpt ~/nla/L40/rl_vllm/critic_latest \
    --sidecar ~/nla/L40/rl_vllm --layer 40 --n-shots 4 \
    --device cuda:0 --ar-device cuda:1

python notebooks/ioi/ioi_text_intervention.py \
    --ar-ckpt ~/nla/L40/rl_vllm/critic_latest --sidecar ~/nla/L40/rl_vllm --layer 40 \
    --explanations notebooks/ioi/results/ioi_verbalizations_4shot_L40.json \
    --archive      notebooks/ioi/results/ioi_patch_conditions_4shot_L40.json \
    --device cuda:0 --ar-device cuda:1
```

`--limit-a 3 --limit-b 3` on pass 2 is a quick smoke test. Both scripts take
`--k-positions K` to patch the last K tokens instead of only the last one; the
published figures use the default, K = 1.

## How the two passes fit together

1. **Pass 1** draws 200 prompts (seed 1234), builds the few-shot prompt, and for
   each prompt records the base model's answer, the AV's explanation of the
   last-token activation, and the AR's reconstruction of it. It then patches
   each of six vectors in and records the answer again. A prompt whose
   explanation never closes its `</explanation>` tag is dropped (0–16 per cell).
2. **Pass 2** reads pass 1's explanations and splits prompts into two groups by
   whether the round trip (`nla_norm`) got the answer right. Group A (right,
   and the explanation names the answer) gets name substitutions; group B
   (wrong) gets the answer name added. Every edited text is re-encoded by the
   AR and patched in the same way as pass 1. Names for the substitution arms are
   drawn from a per-prompt seed, so a rerun reproduces the same archive.
3. **Plot** reads the two archives per cell. Error bars are 95% Wilson
   intervals over prompts; for bars that pool three substitutions per prompt,
   the prompt is the unit, not the draw.

Details of each condition and edit are in the module docstrings, which are
short and meant to be read.

## Notes on the shipped archives

- The archives in `results/` were produced by earlier versions of these scripts
  in several passes (a base run, two top-up runs merged in afterwards, and a
  pass that added the `tok1_is_*` fields). The scripts here do all of that in
  one run per cell and produce the same variant keys and the same injected
  names, checked against every group-A row of the shipped archives. The
  `av_lora` and `ar_ckpt` fields name the Hugging Face checkpoints the
  originals were loaded from.
- The shipped text archives also include group-A prompts whose explanation never
  names the answer, with no-op edits. A fresh run skips those prompts, since the
  figure does not draw them; the `n = kept of total` legend counts are unchanged.
- Layer 24's panel A is empty at every shot count: that AV never writes the
  answer name into its explanations, so there is nothing to substitute. That is
  a result, not a missing run.
