#!/usr/bin/env bash
# Reproducible Python env for kernel-bench on the OIST saion cluster.
#
# Why micromamba: saion has no system python3.11 and no conda/micromamba on PATH (only ~/.local has
# torch+triton). micromamba is a single static binary (no root) that provides a clean python 3.11,
# so the env is reproducible and independent of the login node's ad-hoc state.
#
# Run on the saion LOGIN node (pip can download there). Then VERIFY on a COMPUTE node: login nodes
# have old GLIBC (<2.28) and fail `import torch` -- it only loads on the A100 nodes.
#
# Usage:
#   bash scripts/setup_cluster_env.sh [ENV_PREFIX]
#     ENV_PREFIX (default /work/SzollosiU/enzo-marsot/kbench-env) -- scratch: fast but may be purged.
#     Pass a /bucket path to build a PERSISTENT env (login node has /bucket rw); reuse it every run.
set -euo pipefail

ENV_PREFIX="${1:-/work/SzollosiU/enzo-marsot/kbench-env}"
MAMBA_ROOT="${MAMBA_ROOT:-/work/SzollosiU/enzo-marsot/micromamba}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ="$REPO_ROOT/requirements.txt"
# torch wheel index for the A100 nodes' CUDA (cu130). Override TORCH_INDEX for a different CUDA.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

# 1. micromamba (static binary, no root) if not present
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
MM="$MAMBA_ROOT/bin/micromamba"
if [ ! -x "$MM" ]; then
  echo ">> installing micromamba into $MAMBA_ROOT"
  mkdir -p "$MAMBA_ROOT"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MAMBA_ROOT" bin/micromamba
fi

# 2. python 3.11 env
echo ">> creating python 3.11 env at $ENV_PREFIX"
"$MM" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip

# 3. deps: torch (cu130) first from its index, then triton/numpy/scipy
echo ">> installing torch from $TORCH_INDEX, then requirements"
"$ENV_PREFIX/bin/pip" install --index-url "$TORCH_INDEX" torch
"$ENV_PREFIX/bin/pip" install -r "$REQ"
# install the package itself (editable) so `import kbench` / `newton` resolve
"$ENV_PREFIX/bin/pip" install -e "$REPO_ROOT" --no-deps

cat <<EOF

>> env ready at $ENV_PREFIX
>> VERIFY ON A COMPUTE NODE (login nodes fail torch import -- GLIBC):
   srun -p gpu-a100 -c 4 --mem=16G --gres=gpu:1 --time=00:05:00 \\
     $ENV_PREFIX/bin/python -c 'import torch,triton,numpy,scipy; \\
       print("torch",torch.__version__,"cuda",torch.cuda.is_available())'
EOF
