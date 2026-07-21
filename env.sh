# EasyNLA environment — source this before training.
#
# CUDA forward compatibility
# --------------------------
# Kernel-mode driver here is 535.309.01 (CUDA 12.2 branch). The container ships a
# CUDA 12.6 forward-compat userspace at /usr/local/cuda-12.6/compat (libcuda
# 560.35.05), which ldconfig already prefers. cu128 wheels do run on that via CUDA
# minor-version compatibility, but vLLM exercises the VMM / IPC driver APIs for its
# weight sync, so we pin the exact-matching CUDA 12.8 compat userspace
# (libcuda 570.211.01, from cuda-compat-12-8) ahead of it.
#
# Installed user-local (no root) from
#   developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
#     cuda-compat-12-8_570.211.01-0ubuntu1_amd64.deb
export LD_LIBRARY_PATH="$HOME/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export PATH="$HOME/.local/bin:$PATH"

# The IPC weight sync in train_rl_vllm needs the legacy CUDA allocator —
# expandable_segments breaks it (see README).
unset PYTORCH_CUDA_ALLOC_CONF

# vLLM v1 runs EngineCore in a subprocess; the IPC weight sync ships a
# functools.partial through apply_model, which msgspec refuses without this
# (matches scripts/smoke_sync_rl.sh).
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

export NLA_REPO=$HOME/EasyNLA
export NLA_DATA=$HOME/nla/data
export NLA_CKPTS=$HOME/nla/ckpts
export VLLM_LENS_VENV=${VLLM_LENS_VENV:-$HOME/envs/vllm-lens}
