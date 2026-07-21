# Agent runbook — train an NLA on Qwen2.5-7B-Instruct (8×A100-80GB)

Written for an agent picking this up cold. It is the end-to-end path that actually
worked, including the failures worth not repeating. Reproduces:

| Stage | Metric | Expected |
|---|---|---|
| AV SFT | held-out val perplexity | **~4.07** (from ~69 at step 0) |
| AR SFT | held-out FVE (gold explanations) | **~62%** |
| RL step 0 | held-out FVE (own explanations) | **~56%** |
| RL step 400 | held-out FVE | **~73%** |

Wall clock on 8×A100-80GB: activations ~40 min, SFT ~1 h 40 m, RL ~7 h 10 m.

**Verify each stage against those numbers before proceeding.** Every failure mode
below was silent — the run trains happily and produces a worse model.

---

## 0. Hardware / driver assumptions

Written for 8×A100-80GB. The repo's tuned defaults assume **H200 (141 GB)** and
will OOM on 80 GB — see §5. If you are on H200s, drop the memory workarounds.

CUDA forward compatibility: the kernel driver here is 535.309.01 (CUDA 12.2
branch) but the wheels are cu128. A CUDA 12.8 forward-compat userspace
(`libcuda.so.570.211.01`, from `cuda-compat-12-8`) is installed user-local at
`~/cuda-compat` and put on `LD_LIBRARY_PATH` by `env.sh`. cu128 wheels *do* also
run on the container's older 12.6 compat layer via CUDA minor-version
compatibility, but vLLM exercises the VMM/IPC driver APIs for its weight sync, so
the exact-matching 12.8 userspace is what we pin.

**`source env.sh` before every command.** It sets that `LD_LIBRARY_PATH`, sets
`VLLM_ALLOW_INSECURE_SERIALIZATION=1`, and *unsets* `PYTORCH_CUDA_ALLOC_CONF`.

---

## 1. Environments (one-time)

Two venvs, deliberately:

```bash
# main — SFT, data prep, plots
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e . --torch-backend=cu128
uv pip install --python .venv/bin/python bitsandbytes matplotlib

# rollout — RL only (pins vllm==0.19.0)
bash scripts/install_vllm_lens.sh          # -> ~/envs/vllm-lens, applies the patch
```

### Two upstream bugs are already fixed in this repo — do not revert them

1. **`utils/patch_vllm_lens.py`** validated every hunk against the *pristine*
   source, but hunks 9 and 12 anchor on text that hunks 7 and 8 introduce. A fresh
   venv therefore always refused at hunk 9; only *upgrading an already-patched*
   venv worked. Now it matches against progressively-patched text, so `HUNKS` must
   stay in dependency order. Expect `already patched (all 16 hunks)`.

2. **`peft` is pinned to `0.18.1`** in `install_vllm_lens.sh`. peft 0.19.x's
   `set_peft_model_state_dict` unconditionally imports `EmbeddingParallel` from
   `transformers.integrations.tensor_parallel`, which does not exist in the pinned
   transformers 4.57.1. It is guarded by `torch.distributed.is_initialized()`, so
   single-process `merge_lora_to_hf.py` is unaffected but **every `torchrun`
   `train_rl_vllm` run with `--av-adapter` dies at startup on all ranks.**

Sanity check the rollout venv:

```bash
~/envs/vllm-lens/bin/python utils/patch_vllm_lens.py     # "already patched (all 16 hunks)"
~/envs/vllm-lens/bin/python -c "import nla, peft; print(peft.__version__)"   # 0.18.1
```

---

## 2. Data — the published dataset has **no activations**

`ceselder/qwen3-8b-nla-L24-finefineweb-100k` ships `detokenized_text_truncated`
but **not** `activation_vector`. Every trainer reads that column, so the dataset
cannot be trained on as published. Regenerate it.

What is reusable across base models — do **not** re-run datagen, it costs ~250k
Claude Batches API calls:

- **gold explanations** (`response`) — generated from the *text*, model-agnostic
- **prompts** — store the `<INJECT>` placeholder, not a literal marker, so they are
  tokenizer-agnostic (the char is substituted at load time from the sidecar)
- **`n_raw_tokens` and doc splits** — Qwen2.5-7B-Instruct tokenizes this corpus
  *identically* to Qwen3-8B and shares the marker token id (`㈎`, 149705). Verified
  on 3000 sampled rows. Re-verify if you retarget a non-Qwen model.

```bash
huggingface-cli download ceselder/qwen3-8b-nla-L24-finefineweb-100k \
    --repo-type dataset --local-dir ~/nla/data

SRC=~/nla/data OUT=~/nla/data25 MODEL=Qwen/Qwen2.5-7B-Instruct LAYER=20 \
  LOG_NAME=regen25 bash scripts/run_regen_all.sh          # ~40 min, 8 GPUs

.venv/bin/python scripts/make_val_split.py --in ~/nla/data25/av_sft_shuf.full.parquet
.venv/bin/python scripts/make_val_split.py --in ~/nla/data25/ar_sft_shuf.full.parquet
```

### The layer convention is off by one — get this wrong and everything still "works"

`--layer 20` means the **output of block 20**, which equals HF
`hidden_states[21]` (index 0 is the embedding output). This matches
`datagen/extractors.py` and `models.py`. Verify numerically, don't trust it:

```
cos(mine, hidden_states[21]) = 0.9999      <- correct
cos(mine, hidden_states[20]) = 0.914
```

`regenerate_activations.py --layer` also **rewrites the sidecar and the
`activation_layer` column**. That matters: `train_sft` derives the AR critic depth
from `extraction.layer_index + 1`, so a stale sidecar either asserts or silently
trains a wrong-depth critic.

Expected after regeneration: `d_model=3584`, `activation_layer=20`, sidecar
`base_model=Qwen/Qwen2.5-7B-Instruct`, ~0.003% of rows dropped (their text ends
mid-multibyte-character, so it cannot reproduce the original token boundary — they
are dropped, never repositioned). Val splits: 245,344 train / 1,911 val (AV).

The split is **doc-disjoint** via `val_split.is_val_doc` (crc32 on `doc_id`). Do
not substitute a row-index split: the corpus is row-shuffled and each doc
contributes ~10 rows, so a row split leaves ~zero fully-unseen docs.

---

## 3. SFT — 8-GPU data-parallel

```bash
DATA=~/nla/data25 CKPT=~/nla/ckpts25 BASE=Qwen/Qwen2.5-7B-Instruct \
  LOG_NAME=sft25 bash scripts/run_sft_all.sh              # ~1 h 40 m
```

`train_sft.py` **gained DDP in this repo** (it is single-GPU upstream). Notes:

- `--batch-size` is the **global** batch, split across ranks — so effective batch
  and the tuned LR are invariant to GPU count. It must divide `world_size`.
- Gradients are averaged with a manual `all_reduce`, not `DistributedDataParallel`
  (DDP's hook bucketing interacts badly with the AV path's gradient checkpointing +
  re-firing injection hook + mostly-frozen LoRA params). Same approach as
  `train_rl_vllm`.
- All ranks seed identically **after** dist init — the manual all-reduce only keeps
  replicas in sync if they *start* identical, and LoRA A / the AR value head are
  randomly initialized.
- Rank 0 owns all side effects (stdout, wandb, checkpoints, held-out evals).
- Verified equivalent to single-GPU: same seed → losses agree to ~0.003 and
  response-token counts match *exactly* (proving the shards are disjoint and
  complete). ~7× faster (1.25 s/step vs 8.9 s).

Expect in the log: `--ar-num-layers defaulted to 21 (sidecar layer_index+1)`. If it
says anything else, the sidecar is wrong — stop.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the SFT script and is
**safe here**; only the RL trainer's CUDA-IPC weight sync requires the legacy
allocator. Without it, the repo's default `--batch-size 64` OOMs on 80 GB.

---

## 4. Merge, and validate the merge

RL needs merged bf16 checkpoints (`NLACriticModel.from_pretrained` cannot read the
AR LoRA + value-head format).

```bash
.venv/bin/python scripts/merge_lora_to_hf.py \
  --base-ckpt Qwen/Qwen2.5-7B-Instruct \
  --av-dir ~/nla/ckpts25/av_sft/iter_0003834 \
  --ar-dir ~/nla/ckpts25/ar_sft/iter_0003834 \
  --av-out ~/nla/ckpts25/merged/av_hf --ar-out ~/nla/ckpts25/merged/ar_hf --mode both
```

**Then check the merged AR's held-out FVE equals the LoRA checkpoint's** — the
script's own docstring requires this, and the merge is the step most likely to
silently drop the value head. Expect FVE to match to ~0.1% (we got 62.0% / mse
0.2763 both sides). A `lm_head`/`model.norm` "newly initialized" warning is
expected and harmless: the critic strips the final norm and uses `value_head`.

---

## 5. RL — memory is the whole problem on 80 GB

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct DATA=~/nla/data25 CKPT=~/nla/ckpts25 \
  LOG_NAME=rl25 bash scripts/run_rl.sh                    # ~7 h 10 m
# resume:  RESUME=~/nla/ckpts25/rl_vllm/iter_000150 ... bash scripts/run_rl.sh
```

The repo header budgets ~115 GB peak on H200. On 80 GB you must cut ~36 GB. Tuned
settings, measured:

| config | step time | peak |
|---|---|---|
| repo default (GC on, mb 4, vLLM 0.30) — Qwen3-8B | 77 s | 71–77 GB |
| no GC, mb 8, vLLM 0.28 | **OOM** | 79–81 GB |
| GC on, mb 8, vLLM 0.28 | 68 s | 68–72 GB |
| **no GC, mb 4, vLLM 0.24** (what `run_rl.sh` uses) | **62 s** | 72–75 GB |

The lever that makes this work is **KV geometry**: Qwen2.5-7B-Instruct uses
**56 KB/token** (4 KV heads × 28 layers) vs Qwen3-8B's **144 KB** (8 × 36). So vLLM
runs fine at `--vllm-gpu-mem 0.24` where Qwen3 needs 0.30, and the memory that
frees pays for dropping gradient checkpointing (~30% of training compute).
Floor: the weights alone are ~15.2 GB, so `--vllm-gpu-mem` below ~0.20 leaves no
KV at all. **Recompute this for any new base model — do not copy 0.24 blindly.**

Also required, and already handled by `run_rl.sh`:

- `--ar-lora` — co-train the AR as LoRA instead of full fine-tune (~20 GB). The
  repo notes the merged AR is the frozen base and a zero-init LoRA starts identical
  to it, so the warm-start is unchanged.
- `--av-adapter <av_sft adapter>` — continues the SFT adapter and keeps a frozen
  copy as the KL reference. Without it RL starts from a fresh zero-init LoRA on the
  merged AV: a measured **~12 pp FVE cold-start**. `--av-ckpt` is still required
  (it feeds the vLLM engine + tokenizer).
- `--evals base_fve` — the config defaults to `[base_fve, text_judges]` and
  `text_judges` needs `ANTHROPIC_API_KEY`. Without a key, **`ext%` in the log is
  your main guard against text degradation** — watch it.
- `PYTORCH_CUDA_ALLOC_CONF` **unset** — `expandable_segments` breaks the IPC weight
  sync.

### Health checks at RL step 0

- eval FVE ≈ warm-start level (~56%); `ext 100%`; `kl` ~1e-3 (near-zero proves the
  `--av-adapter` KL reference loaded, not a fresh LoRA)
- generations are coherent English. **CJK output is the repo's signature of failed
  injection.** The patched vllm-lens also asserts per-request steering invariants
  each step, so surviving past step 0 is itself evidence injection fires.

Healthy trajectory: FVE 56.4 → 69.3 (step 100) → 70.9 (200) → 72.4 (300) → 72.9
(400). KL rises smoothly to ~0.95; entropy rises 1.43 → 1.96 (rising entropy with
rising reward is fine here — more varied phrasings that still reconstruct).

---

## 6. Plots

```bash
.venv/bin/python scripts/plot_curves.py \
  --sft-log ~/nla/logs/sft25.log --rl-log ~/nla/logs/rl25.log \
  --out-dir ~/nla/plots --tag "Qwen2.5-7B-Instruct · block 20"
```

Parses the text logs (we run `--no-wandb`). RL per-step lines are printed by all 8
ranks and are averaged — rank 0 alone is a 1/8 sample of the global batch.

---

## 7. Operational gotchas that cost real time

- **Always launch through the `scripts/run_*.sh` wrappers**, never a hand-typed
  command. They all end in `2>&1 | tee -a "$LOG"`. A hand-copied RL launch once
  lost its redirect and ~3 h of per-step metrics went only to a terminal —
  unrecoverable (no tmux; `strace` absent and `ptrace_scope=1` blocks attaching to
  a running process). Use `tee -a`, never `>`: a resumed run must extend the log.
- **Detach properly**: `setsid nohup … &`. Verify with `ps -eo pid,ppid,sess,tty` —
  want `ppid=1`, `tty=?`, and `sess` equal to the pid.
- **`--save-every`**: the config default (100) means an interrupted RL run loses up
  to 100 steps. `run_rl.sh` uses 25. Nothing is resumable before the first save.
- **Never combine `pkill -f <pattern>` with a launch command containing the same
  pattern** — `pkill` matches your own shell's command line and kills the shell
  before it launches, silently. (Bit us twice; the `[t]` bracket trick does *not*
  help if the launch text is in the same command.) Kill in a separate call, or by PID.
- **Filtered pipelines block-buffer.** `python … | grep … | tee` shows nothing for
  minutes and a healthy run looks hung. Use `python -u` and a plain `tee`.
- GPUs read 0% during each regeneration split's CPU-only tokenization pre-pass.
  That is normal, not a hang.

---

## 8. What is checked in

| Path | Purpose |
|---|---|
| `env.sh` | CUDA forward-compat `LD_LIBRARY_PATH`, vLLM env, allocator hygiene |
| `scripts/regenerate_activations.py` | rebuild `activation_vector`; `--layer` retargets model/depth and rewrites the sidecar |
| `scripts/make_val_split.py` | doc-disjoint train/val split via `val_split.is_val_doc` |
| `scripts/run_regen_all.sh` | resume-safe extraction driver (`SRC/OUT/MODEL/LAYER`) |
| `scripts/run_sft_all.sh` | AV+AR SFT, 8-GPU (`DATA/CKPT/BASE`) |
| `scripts/run_rl.sh` | RL with tuned memory settings and guaranteed logging |
| `scripts/plot_curves.py` | PNG curves from the logs |
| `nla/train_sft.py` | **modified**: DDP support (see §3) |
| `utils/patch_vllm_lens.py` | **modified**: hunk-ordering fix (see §1) |
| `scripts/install_vllm_lens.sh` | **modified**: `peft==0.18.1` pin (see §1) |

Trained artifacts: <https://huggingface.co/Yooniel/qwen2.5-7b-instruct-nla-L20>
