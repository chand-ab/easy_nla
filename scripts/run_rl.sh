#!/bin/bash
# RL (GRPO, vLLM rollouts) — 8-GPU data-parallel, with the log ALWAYS captured.
#
# This script exists so the launch command is never hand-retyped. A previous run
# was started from a copy-pasted command whose `> log 2>&1` had been dropped;
# training was fine but ~3 h of per-step metrics went only to a terminal and were
# unrecoverable (no tmux, and ptrace_scope=1 blocks attaching to capture later).
# `tee -a` here APPENDS, so a resumed run extends the same log instead of
# truncating it — `>` would wipe the earlier history on every relaunch.
#
#   MODEL=Qwen/Qwen2.5-7B-Instruct DATA=.../data25 CKPT=.../ckpts25 \
#   LOG_NAME=rl25 bash scripts/run_rl.sh
#   # resume once a checkpoint exists:
#   RESUME=.../ckpts25/rl_vllm/iter_000150 ... bash scripts/run_rl.sh
#
# Watch it:  tail -f $HOME/nla/logs/<LOG_NAME>.log
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
# PYTORCH_CUDA_ALLOC_CONF must stay UNSET: the CUDA-IPC weight sync needs the
# legacy allocator (expandable_segments breaks it). env.sh unsets it; this is a
# belt-and-braces guard in case the caller re-exported it.
unset PYTORCH_CUDA_ALLOC_CONF

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA=${DATA:-$HOME/nla/data25}
CKPT=${CKPT:-$HOME/nla/ckpts25}
SAVE=${SAVE:-$CKPT/rl_vllm}
STEPS=${STEPS:-400}
RESUME=${RESUME:-}
LOGDIR=$HOME/nla/logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${LOG_NAME:-rl}.log"

# Tuned on 8xA100-80GB for Qwen2.5-7B-Instruct: 62 s/step. Its KV cache is
# 56 KB/token (4 KV heads x 28 layers) vs Qwen3-8B's 144 KB, so vLLM runs fine at
# 0.24 instead of 0.30, and the memory that frees pays for dropping gradient
# checkpointing (~30% of training compute). Peak ~75 GB — for Qwen3-8B use
# --gradient-checkpointing and --vllm-gpu-mem 0.30 instead.
RESUME_ARG=""; [ -n "$RESUME" ] && RESUME_ARG="--resume-from-lora $RESUME"

{
  echo "=== RL started $(date -Is) | model=$MODEL data=$DATA save=$SAVE resume=${RESUME:-<none>} ==="
  ${VLLM_LENS_VENV:-$HOME/envs/vllm-lens}/bin/torchrun --standalone --nproc_per_node=8 \
    -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
    --base-ckpt "$MODEL" \
    --av-ckpt "$CKPT/merged/av_hf" \
    --av-adapter "$CKPT/av_sft/iter_0003834" \
    --ar-ckpt "$CKPT/merged/ar_hf" \
    --rl-parquet "$DATA/rl_shuf.full.parquet" \
    --sidecar "$DATA/rl_shuf.full.parquet" \
    --save-dir "$SAVE" $RESUME_ARG \
    --evals base_fve --num-steps "$STEPS" --batch-prompts 256 --group-size 8 \
    --ar-lora --logp-micro-batch 4 --critic-micro-batch 4 --vllm-gpu-mem 0.24 \
    --eval-every 10 --eval-n-prompts 128 --save-every 25 --no-wandb
  echo "=== RL exited rc=$? $(date -Is) ==="
} 2>&1 | tee -a "$LOG"
