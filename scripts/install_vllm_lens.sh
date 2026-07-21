#!/bin/bash
# Build the vllm-lens venv (fast vLLM RL rollout backend for train_rl_vllm.py).
# Rationale, version-pin reasoning, and the failure modes these pins avoid are in
#   docs/vllm-lens-setup.md
# The injection patch is applied automatically at the end (re-run after rebuilds).
#
# DO NOT bump vllm/torch-backend blindly — see the doc first.
set -euo pipefail
VENV=${1:-$HOME/envs/vllm-lens}

uv venv "$VENV" --python 3.12
# vllm==0.19.0 + vllm-lens==1.1.0: the matched pair where the injection hook
#   fires (vLLM 0.22+ refactored GPUModelRunner -> hook silently no-ops).
# --torch-backend=cu128: targets CUDA 12.8 (driver >=570); the default cu130
#   wheel needs driver >=580 and fails at import on older drivers.
# transformers==4.57.1 (repo-wide pin): vllm's resolver otherwise pulls v5,
#   whose apply_chat_template break crashes the trainers. peft/bitsandbytes/wandb
#   are needed by nla.train_rl_vllm itself (this venv runs the trainer).
# peft==0.18.1: peft 0.19.x's set_peft_model_state_dict unconditionally imports
#   EmbeddingParallel from transformers.integrations.tensor_parallel, which does
#   not exist in the pinned transformers 4.57.1 -> ImportError. It is guarded by
#   `torch.distributed.is_initialized()`, so it only fires on DISTRIBUTED adapter
#   loads: single-process merge_lora_to_hf.py is unaffected, but every torchrun
#   train_rl_vllm run with --av-adapter (the README's recommended warm-start)
#   dies at startup on all ranks. 0.18.1 is the newest peft without that import.
uv pip install --python "$VENV/bin/python" \
  "vllm==0.19.0" "vllm-lens==1.1.0" \
  "transformers==4.57.1" "peft==0.18.1" "bitsandbytes" "wandb" \
  --torch-backend=cu128

echo "=== verify (imports vllm._C -> exercises libcudart) ==="
"$VENV/bin/python" - <<'PY'
import torch, vllm, vllm_lens
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector
print(f"OK — torch {torch.__version__} (cuda {torch.version.cuda}) | "
      f"vllm {vllm.__version__} | vllm_lens {vllm_lens.__version__}")
PY

echo "=== apply the vllm-lens injection patch (REQUIRED — unpatched = weak injection) ==="
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
"$VENV/bin/python" "$REPO_DIR/utils/patch_vllm_lens.py" || {
  echo "FATAL: patch_vllm_lens failed — venv would run UNPATCHED (partial-residual"
  echo "norm-match bug: injection far too weak -> high clip-frac -> divergence)."
  exit 1
}
echo "patched. NOTE: re-run this script (or the patcher) after ANY venv rebuild."
