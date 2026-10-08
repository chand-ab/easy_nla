# Paraphrase and identity: what the ARs read, and whether the arms differ

This folder reproduces `figure_results/paraphrase_identity.svg`. The figure
covers four NLAs trained on Qwen2.5-7B-Instruct at layer 20, each changing one
source of training randomness at a time:

| pair | what differs |
|---|---|
| NLA 1 vs NLA 2 | rollout sampling only (same SFT, same RL data order) |
| NLA 3 vs NLA 1/2 | SFT LoRA init, plus rollout sampling |
| NLA 4 vs NLA 1/2 | RL data order, plus rollout sampling |

Both panels use the same 127 held-out prompts (1 of 128 was dropped for a failed
extraction) and the same layer.

**Left: the AV/AR swap matrix.** Cell (i, j) is held-out FVE when AR_j
reconstructs the activation from AV_i's explanation (greedy decode). The four
original columns are nearly flat, so any AR reads any AV almost as well as its
own partner does (about 0.7 pt own-partner edge). The fifth column scores AV_i's
explanations, rewritten by `google/gemma-3-27b-it` (medium paraphrase with a
fidelity constraint), with AV_i's own AR. Most of the FVE survives the rewrite,
so the ARs are mostly reading content, not a private surface code.

**Right: can a probe tell the arms apart?** For each pair of arms, an L2
logistic regression is trained on the base model's mean-pooled layer-20
activations of the two arms' explanations, using nested 5-fold CV grouped by
prompt. Chance is 50%, and every pair scores 89–96%.

Read together: the arms are clearly different, but the difference is not in
what the AR reads.

## What is here

| file | what it does |
|---|---|
| `plot_paraphrase_identity.py` | **draws the figure** from `results/`. Needs only matplotlib and numpy |
| `paraphrase_effect.py` | loads `matrix_*.json`, picks the rows valid in every cell, and builds the FVE matrices; with `--matrix` it also prints doc-clustered bootstrap CIs |
| `plot_paraphrase.py` | the five-condition swap-matrix figure (`paraphrase_swap_matrices.svg`); also supplies the condition list the main figure uses |
| `plot_swap_matrix.py` | shared colours and styling, plus the temp-0 / temp-1 swap-matrix figures |
| `eval_matrix.py` | GPU: generates the AV explanations (`--phase generate`) and scores every AV × AR cell (`--phase score`). Writes `matrix.json` |
| `paraphrase_generations.py` | GPU: rewrites the cached explanations with the paraphraser, laid out so `eval_matrix.py --phase score` reads them directly |
| `collect_explanation_activations.py` | GPU: runs the explanations through the raw base model and saves mean- and last-token activations at every layer (`.npz`) |
| `plot_activation_logreg.py` | the pairwise identity probe (needs scikit-learn). Archives `activation_logreg_<pool>.json`, which the right panel reads |

`results/` holds the archives behind the published figure:

- `matrix_temp0.json`: the original swap matrix, with per-row MSE for every cell.
- `matrix_temp0_para_{medium,heavy}[_faithful].json`: the same matrix scored on
  paraphrased explanations. The figure uses `medium_faithful`. All five files
  are loaded so the row set matches the five-condition figure.
- `matrix_temp1_seed{0..3}.json`: temperature-1 matrices, used by
  `plot_swap_matrix.py` only.
- `activation_logreg_mean.json`: pairwise probe accuracy at hidden-state
  indices 6, 14 and 21. NLA layer L sits at index L+1, so the figure uses 21.
- `generations/`: the explanations themselves, original (`temp0/`) and
  paraphrased, one `nla<i>.json` per arm. Paraphrased rows keep
  `explanation_orig` and the measured word `overlap`.

The activation `.npz` (about 127 × 4 × 29 × 3584 fp16 values) is not included
because `*.npz` is gitignored. The probe's archived json is enough to redraw.

## Redraw the figure

No GPU and no model download needed. It takes a few seconds.

```bash
pip install matplotlib numpy
python notebooks/paraphrase/plot_paraphrase_identity.py
# -> notebooks/paraphrase/figure_results/paraphrase_identity.svg
```

Pass `--out something.png` for a raster. The SVG is written with a fixed hash
salt and no timestamp, so redrawing from unchanged data produces a
byte-identical file.

The companion figures come from the same data:

```bash
python notebooks/paraphrase/plot_paraphrase.py        # five-condition swap matrices
python notebooks/paraphrase/plot_swap_matrix.py       # temp-0 / temp-1 swap matrices
python notebooks/paraphrase/plot_activation_logreg.py --from-json   # redraw from the archived json
```

## Rerun the experiment

This needs the four trained NLAs (AV checkpoint and RL adapter, plus the AR),
the RL parquet the runs trained on, and GPUs. Install the package from the repo
root with `pip install -e .`, then:

1. **Generate and score the original matrix.** Run one `--phase generate` per NLA
   (one process each, because vLLM doesn't release the GPU between engines).
   Then run a single `--phase score` with all four `--nla` specs:
   ```bash
   python notebooks/paraphrase/eval_matrix.py --phase generate --eval-temperature 0 \
       --nla name=nla1,av_ckpt=...,av_adapter=...,ar_ckpt=... \
       --rl-parquet ... --out-dir runs/temp0          # repeat for nla2..nla4
   python notebooks/paraphrase/eval_matrix.py --phase score --eval-temperature 0 \
       --nla name=nla1,... --nla name=nla2,... --nla name=nla3,... --nla name=nla4,... \
       --rl-parquet ... --out-dir runs/temp0
   ```
   This writes `runs/temp0/generations/nla<i>.json` and `runs/temp0/matrix.json`.
   The archive stores them as `results/generations/temp0/` and
   `results/matrix_temp0.json`.
   Validate first: `--only-diagonal` at temperature 1 must reproduce each run's
   own last in-run eval. See the docstring.
2. **Paraphrase and re-score.** Follow the commands in the
   `paraphrase_generations.py` docstring. Then score each
   `temp0_para_<level>` directory with `eval_matrix.py --phase score`.
3. **Activations and probe.**
   ```bash
   python notebooks/paraphrase/collect_explanation_activations.py \
       --gen-dir notebooks/paraphrase/results/generations/temp0 \
       --out notebooks/paraphrase/results/activations/greedy_base_subset.npz
   python notebooks/paraphrase/plot_activation_logreg.py     # ~25 min, nested CV
   ```
4. Redraw as above.
