#!/usr/bin/env bash
# §5.1 speed comparison: gpurec genewise DTL-rate fitting on the SAME archaea >=4-species family
# subset AleRax uses (archaea_ge4sp.families.txt). Reports the warm fit time + (optionally) the
# certified optimum, for the head-to-head against run_alerax.sh.
#
# The optimized recipe is BAKED into run_gpurec_traced.py's defaults:
#   clade_budget=900k (GPU occupancy), Adam 5 steps/lr=1/grad-clip-10 warmup, forward-difference
#   FD-Hessian, eager-defer of cert-failing tail families, PD cert OFF.  See OPTIMIZATION_PLAN.md.
# This wrapper only supplies the machine-specific data/library paths + env; no recipe knobs needed.
#
# Warm timing: run once to warm the Triton kernel cache (~/.triton), then again for the reported number.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- machine-specific paths (override via env if your checkout differs) -----------------------------
# Worktree holding the gpurec library, run_cv.py (DATASETS/_CV_SO), and the Rust preprocess .so:
GPUREC_WT="${GPUREC_WT:-/home/enzo/Documents/git/gpurec/agent-worktrees/kernel-bench-mapcv-merge}"
PYTHON="${PYTHON:-/home/enzo/miniforge3/bin/python}"
PREPROCESS_SO="${PREPROCESS_SO:-$GPUREC_WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so}"
FAMILIES="$HERE/archaea_ge4sp.families.txt"

# (re)build the families file so the run is self-contained / reproducible (same set AleRax uses)
"$PYTHON" "$HERE/make_families.py"

# CERT=1  -> also run the optional pi64/neu64 PD certificate over all families (+~45s; off by default).
# MICRO=1 -> per-operation timing breakdown.  See run_gpurec_traced.py for the full env knob list.
echo "=== gpurec genewise run @ $(date -Is) | host=$(hostname) | CERT=${CERT:-0} ==="
CERT="${CERT:-0}" \
FAMFILE="$FAMILIES" DATASET=archaea \
PYTHONPATH="$GPUREC_WT:$GPUREC_WT/experiments/sanderson_cv" \
GPUREC_PREPROCESS_PATH="$PREPROCESS_SO" PYTHONNOUSERSITE=1 \
"$PYTHON" -u "$HERE/run_gpurec_traced.py" 2>&1 \
  | grep --line-buffered -vE 'it/s|%\|' | tee "$HERE/run_gpurec.log"
echo "=== DONE @ $(date -Is) ==="
