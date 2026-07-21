#!/bin/bash
# Regenerate the activation_vector column for every split, detached, with a
# live log. See scripts/regenerate_activations.py for why this is needed at all
# (the published dataset ships text only).
#
#   bash scripts/run_regen_all.sh            # foreground
#   setsid nohup bash scripts/run_regen_all.sh >/dev/null 2>&1 &   # detached
#
# Watch it:  tail -f $HOME/nla/logs/regen.log
#
# python -u and a plain tee (no grep in the pipeline) — a filtered pipeline
# block-buffers, so progress only appeared in ~4 KB bursts and a healthy run
# looked hung.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

# Overridable so the same script retargets a different base model / layer:
#   SRC=... OUT=... MODEL=Qwen/Qwen2.5-7B-Instruct LAYER=20 LOG_NAME=regen25 bash scripts/run_regen_all.sh
SRC=${SRC:-$HOME/nla/data}          # where the source parquets live
OUT=${OUT:-$HOME/nla/data}          # where the regenerated ones go
MODEL=${MODEL:-Qwen/Qwen3-8B}
LAYER=${LAYER:-}                            # empty => keep the dataset's own layer
LOGDIR=$HOME/nla/logs
mkdir -p "$LOGDIR" "$OUT"
LOG="$LOGDIR/${LOG_NAME:-regen}.log"
LAYER_ARG=""; [ -n "$LAYER" ] && LAYER_ARG="--layer $LAYER"

{
  echo "=== regen started $(date -Is) pid=$$ | model=$MODEL layer=${LAYER:-<from data>} out=$OUT ==="
  for f in av_sft_shuf ar_sft_shuf rl_shuf; do
    out="$OUT/$f.full.parquet"
    # Resume-safe: a completed split is left alone, so an interrupted run can be
    # relaunched with the same command without redoing finished work.
    if [ -s "$out" ]; then
      echo "########## $f — already done ($(stat -c%s "$out") bytes), skipping ##########"
      continue
    fi
    echo "########## $f — starting $(date -Is) ##########"
    rm -rf "$out.shards"
    .venv/bin/python -u scripts/regenerate_activations.py \
      --in "$SRC/$f.parquet" --out "$out" \
      --base-model "$MODEL" $LAYER_ARG --gpus 8
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "########## $f FAILED rc=$rc — aborting $(date -Is) ##########"
      exit $rc
    fi
    echo "########## $f — done $(date -Is) ##########"
  done
  echo "=== ALL SPLITS DONE $(date -Is) ==="
  ls -la "$OUT"/*.full.parquet
} 2>&1 | tee -a "$LOG"
