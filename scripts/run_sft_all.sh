#!/bin/bash
# Full AV + AR warm-start SFT, 8-GPU data-parallel, one epoch each.
#
#   setsid nohup bash scripts/run_sft_all.sh >/dev/null 2>&1 &
#   tail -f $HOME/nla/logs/sft.log
#
# Sequential, not concurrent: each stage uses all 8 GPUs, and AR is only ~20 min.
# --batch-size is the GLOBAL batch split across ranks, so eff_batch (and the
# tuned LR) match a single-GPU run — only wall-clock changes.
#
# expandable_segments is safe HERE (it is only the RL trainer's CUDA-IPC weight
# sync that requires the legacy allocator), and it is what lets batch 64 fit.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Overridable so the same script trains a different base model:
#   DATA=.../data25 CKPT=.../ckpts25 BASE=Qwen/Qwen2.5-7B-Instruct \
#   LOG_NAME=sft25 bash scripts/run_sft_all.sh
DATA=${DATA:-${NLA_DATA:-$HOME/nla/data}}
CKPT=${CKPT:-${NLA_CKPTS:-$HOME/nla/ckpts}}
BASE=${BASE:-Qwen/Qwen3-8B}
LOGDIR=$HOME/nla/logs
mkdir -p "$LOGDIR" "$CKPT"
LOG="$LOGDIR/${LOG_NAME:-sft}.log"

TORCHRUN=".venv/bin/torchrun --standalone --nproc_per_node=8"

{
  echo "=== SFT started $(date -Is) pid=$$ | base=$BASE data=$DATA ckpt=$CKPT ==="

  # ---- AV: verbalizer, LoRA on the full bf16 base (no 4-bit: merging a 4-bit
  # -trained adapter into bf16 for RL is lossy, and 80 GB has the headroom). ----
  if [ -d "$CKPT/av_sft" ] && ls "$CKPT/av_sft"/iter_* >/dev/null 2>&1; then
    echo "########## AV — checkpoints already exist, skipping ##########"
  else
    echo "########## AV SFT — starting $(date -Is) ##########"
    $TORCHRUN -m nla.train_sft \
      --mode av --base-ckpt "$BASE" \
      --parquet "$DATA/av_sft_shuf.full.train.parquet" \
      --sidecar "$DATA/av_sft_shuf.full.train.parquet" \
      --heldout-parquet "$DATA/av_sft_shuf.full.val.parquet" \
      --save-dir "$CKPT/av_sft" \
      --use-lora --lr 1e-4 --batch-size 64 \
      --heldout-every 500 --save-every 1000 --sample-every 1000 \
      --no-wandb || { echo "########## AV FAILED rc=$? ##########"; exit 1; }
    echo "########## AV SFT — done $(date -Is) ##########"
  fi

  # ---- AR: reconstructor. --heldout-parquet must be an AV-split file: the
  # held-out FVE is scored on (explanation, activation) pairs and only the AV
  # schema carries `response`. Both splits were cut with the same is_val_doc
  # bucket, so those docs are absent from AR's training rows too. ----
  if [ -d "$CKPT/ar_sft" ] && ls "$CKPT/ar_sft"/iter_* >/dev/null 2>&1; then
    echo "########## AR — checkpoints already exist, skipping ##########"
  else
    echo "########## AR SFT — starting $(date -Is) ##########"
    $TORCHRUN -m nla.train_sft \
      --mode ar --base-ckpt "$BASE" \
      --parquet "$DATA/ar_sft_shuf.full.train.parquet" \
      --sidecar "$DATA/ar_sft_shuf.full.train.parquet" \
      --heldout-parquet "$DATA/av_sft_shuf.full.val.parquet" \
      --save-dir "$CKPT/ar_sft" \
      --use-lora --lr 2e-5 --batch-size 64 \
      --heldout-every 500 --save-every 1000 \
      --no-wandb || { echo "########## AR FAILED rc=$? ##########"; exit 1; }
    echo "########## AR SFT — done $(date -Is) ##########"
  fi

  echo "=== SFT ALL DONE $(date -Is) ==="
} 2>&1 | grep -vE "Loading checkpoint shards|torch_dtype. is deprecated|use_cache=True" | tee -a "$LOG"
