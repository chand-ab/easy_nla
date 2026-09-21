#!/usr/bin/env bash
# Run the whole IOI sweep: for every layer and shot count, pass 1
# (ioi_intervention.py) and then pass 2 (ioi_text_intervention.py).
#
#   NLA_DIR=~/nla bash notebooks/ioi/run_ioi.sh
#
# NLA_DIR holds one folder per layer, each an `hf download` of
# achand45/gemma-3-12b-it-nla-L<layer> (see README.md):
#
#   $NLA_DIR/L40/rl_vllm/iter_000400     the AV LoRA
#   $NLA_DIR/L40/rl_vllm/critic_latest   the AR
#   $NLA_DIR/L40/rl_vllm/nla_meta.yaml   the sidecar
#
# A finished cell leaves a marker in logs/, and the script skips cells that
# have one, so a killed sweep can simply be restarted. The marker is separate
# from the output file on purpose: both scripts rewrite their archive after
# every row, so an archive that exists is not an archive that is complete.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
NLA_DIR=${NLA_DIR:?set NLA_DIR to the folder holding L24/ L32/ L40/ L47/}
PY=${PY:-python}
LAYERS=${LAYERS:-"24 32 40 47"}
SHOTS=${SHOTS:-"0 2 4 6 8"}
DEV=${DEV:-cuda:0}          # base model
AR_DEV=${AR_DEV:-cuda:1}    # AR critic; both fit one 80 GB card if you set AR_DEV=$DEV
RESULTS=$HERE/results
LOGS=$HERE/logs
mkdir -p "$RESULTS" "$LOGS"

for L in $LAYERS; do
  NLA=$NLA_DIR/L$L/rl_vllm
  for f in "$NLA/iter_000400/adapter_config.json" "$NLA/critic_latest" "$NLA/nla_meta.yaml"; do
    [ -e "$f" ] || { echo "missing $f — download achand45/gemma-3-12b-it-nla-L$L first"; exit 1; }
  done
done

for L in $LAYERS; do
  NLA=$NLA_DIR/L$L/rl_vllm
  for S in $SHOTS; do
    cell="${S}shot_L${L}"
    PATCH=$RESULTS/ioi_patch_conditions_$cell.json
    EXPL=$RESULTS/ioi_verbalizations_$cell.json

    if [ -e "$LOGS/$cell.patch.done" ]; then
      echo "[skip] pass 1 $cell"
    else
      echo "=== pass 1 $cell"
      if ! "$PY" "$HERE/ioi_intervention.py" \
            --av-lora "$NLA/iter_000400" --ar-ckpt "$NLA/critic_latest" \
            --sidecar "$NLA" --layer "$L" --n-shots "$S" \
            --device "$DEV" --ar-device "$AR_DEV" \
            2>&1 | tee "$LOGS/$cell.patch.log"; then
        echo "pass 1 failed on $cell — see $LOGS/$cell.patch.log"; exit 1
      fi
      touch "$LOGS/$cell.patch.done"
    fi

    if [ -e "$LOGS/$cell.text.done" ]; then
      echo "[skip] pass 2 $cell"
    else
      echo "=== pass 2 $cell"
      if ! "$PY" "$HERE/ioi_text_intervention.py" \
            --ar-ckpt "$NLA/critic_latest" --sidecar "$NLA" --layer "$L" \
            --explanations "$EXPL" --archive "$PATCH" \
            --device "$DEV" --ar-device "$AR_DEV" \
            2>&1 | tee "$LOGS/$cell.text.log"; then
        echo "pass 2 failed on $cell — see $LOGS/$cell.text.log"; exit 1
      fi
      touch "$LOGS/$cell.text.done"
    fi
  done
done

echo "sweep complete — now: $PY $HERE/plot_ioi_grids.py"
