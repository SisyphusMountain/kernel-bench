#!/usr/bin/env bash
# Reproducible Python env for kernel-bench on the OIST saion cluster.
#
# BUILDS ON A COMPUTE NODE (glibc 2.28 + internet): there the latest torch/triton/numpy/scipy install
# as manylinux_2_28 wheels -- no pins, no source build. The login node (glibc 2.17, GCC 4.8.5) would
# force failing source builds (numpy: "requires GCC >= 9.3"), so this script re-execs itself onto a
# gpu-a100 node via srun. Self-contained: PYTHONNOUSERSITE=1 hides ~/.local so torch installs fresh.
#
# Usage (from the saion login node):
#   bash scripts/setup_cluster_env.sh [ENV_PREFIX]      # default /work/SzollosiU/enzo-marsot/kbench-env
set -uo pipefail

SCRIPT="$(readlink -f "$0")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
ENV_PREFIX="${1:-/work/SzollosiU/enzo-marsot/kbench-env}"
MAMBA_ROOT="${MAMBA_ROOT:-/work/SzollosiU/enzo-marsot/micromamba}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/work/SzollosiU/enzo-marsot/pip-cache}"

# --- re-exec on a compute node (glibc 2.28) so modern wheels install -------------------------
if [ "${KBENCH_ON_COMPUTE:-0}" != "1" ]; then
  echo ">> building on a gpu-a100 compute node (glibc 2.28 + internet) via srun"
  exec srun -p gpu-a100 -c 8 --mem=32G --gres=gpu:1 --time=00:40:00 \
    env KBENCH_ON_COMPUTE=1 MAMBA_ROOT="$MAMBA_ROOT" TORCH_INDEX="$TORCH_INDEX" \
        PIP_CACHE_DIR="$PIP_CACHE_DIR" \
    bash "$SCRIPT" "$ENV_PREFIX"
fi

# --- worker (runs on the compute node) -------------------------------------------------------
echo ">> building on $(hostname) (glibc $(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')) -> $ENV_PREFIX"
export PYTHONNOUSERSITE=1 MAMBA_ROOT_PREFIX="$MAMBA_ROOT" PIP_CACHE_DIR

MM="$MAMBA_ROOT/bin/micromamba"
if [ ! -x "$MM" ]; then
  echo ">> installing micromamba (static binary, no root)"; mkdir -p "$MAMBA_ROOT"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MAMBA_ROOT" bin/micromamba
fi

echo ">> (re)creating python 3.11 env"
rm -rf "$ENV_PREFIX"
"$MM" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip

PIP="$ENV_PREFIX/bin/pip"
echo ">> installing torch (cu130) fresh into the env"
"$PIP" install --index-url "$TORCH_INDEX" torch
echo ">> installing triton/numpy/scipy + the package (latest manylinux_2_28 wheels)"
"$PIP" install -r "$REPO_ROOT/requirements.txt"
"$PIP" install -e "$REPO_ROOT" --no-deps

echo ">> verify (self-contained; GPU on this node):"
PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" - <<'PY'
import torch, triton, numpy, scipy
print("torch", torch.__version__, "| triton", triton.__version__,
      "| numpy", numpy.__version__, "| scipy", scipy.__version__)
print("torch from:", torch.__file__.split("site-packages/")[-1], "(should be inside the env, not ~/.local)")
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo ">> ENV READY: $ENV_PREFIX  (run python with PYTHONNOUSERSITE=1 to keep it self-contained)"
