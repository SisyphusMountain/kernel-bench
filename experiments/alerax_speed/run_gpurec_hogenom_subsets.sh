#!/usr/bin/env bash
# gpurec genewise rate-fitting wall-clock on the HOGENOM >=4-species subsets, local RTX 4090 -- the
# gpurec side of the AleRax head-to-head. Restricted to families covering >=4 species (the set AleRax
# actually fits, since UndatedDTL silently drops <4-species families), so the comparison is matched:
#     512  -> first 512 of the curated 1055 set, >=4 species  ->  506 families
#     1055 -> all of the curated 1055 set,        >=4 species  -> 1042 families
#     full -> all 12,408 hogenom_full,            >=4 species  -> 10,869 families
#
# RECIPE (default): the FASTER archaea recipe run_gpurec_traced.py (forward-diff Hessian, lean Adam,
# eager-defer) -- ~1.5x faster than the production warm-rebatch driver at the bit-identical optimum
# (verified on hogenom-1055: 221s vs 324s optimize). Override with DRIVER=<...>.
#
# PROTOCOL (per subset): a WARM-UP run first (CERT off; warms the ~/.triton kernel cache, time
# discarded), then a WARM run (CERT on) whose time is reported. The WARM run runs the final cold
# pi64/neu64 PD certificate so we also get certified convergence (n_conv / interior-PD / bound-active)
# and the total NLL. Two numbers are recorded:
#   * fit_s  = the driver's in-process "optimize (incl builds+adam)" time -- the AleRax-comparable
#              quantity (AleRax does no PD cert either);
#   * wall_s = /usr/bin/date wall-clock of the whole warm process -- matches how AleRax is timed
#              (/usr/bin/time -v), the truest apples-to-apples; cert_s/total_s are also recorded.
#
# Run on an IDLE GPU (a co-tenant skews wall-clock). Family files are (re)built deterministically by
# make_hogenom_full_families.py so the run is self-contained. Multi-run total ~25-35 min (the traced
# recipe is fast; full dominates). Set SUBSETS to fewer entries to shorten.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- machine-specific paths (override via env if your checkout differs) -----------------------------
GPUREC_WT="${GPUREC_WT:-/home/enzo/Documents/git/gpurec/agent-worktrees/kernel-bench-mapcv-merge}"
PYTHON="${PYTHON:-/home/enzo/miniforge3/bin/python}"
PREPROCESS_SO="${PREPROCESS_SO:-$GPUREC_WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so}"
DRIVER="${DRIVER:-$HERE/run_gpurec_traced.py}"                 # the faster recipe, by default
FAM1055="$GPUREC_WT/experiments/sanderson_cv/families_1055.txt"

export PYTHONPATH="$GPUREC_WT:$GPUREC_WT/experiments/sanderson_cv"
export GPUREC_PREPROCESS_PATH="$PREPROCESS_SO"
export PYTHONNOUSERSITE=1

# --- preflight --------------------------------------------------------------------------------------
[ -f "$DRIVER" ]        || { echo "ERROR: driver not found: $DRIVER" >&2; exit 1; }
[ -f "$PREPROCESS_SO" ] || { echo "ERROR: preprocess .so not found: $PREPROCESS_SO" >&2; exit 1; }
[ -f "$FAM1055" ]       || { echo "ERROR: families_1055.txt not found: $FAM1055" >&2; exit 1; }

# --- (re)build the >=4-species family files for each subset (deterministic) -------------------------
echo "=== building >=4-species family files ==="
"$PYTHON" "$HERE/make_hogenom_full_families.py" --source-format namelist --source-file "$FAM1055" \
    --count 512 --out "$HERE/hogenom_512_ge4sp.families.txt"
"$PYTHON" "$HERE/make_hogenom_full_families.py" --source-format namelist --source-file "$FAM1055" \
    --count all --out "$HERE/hogenom_1055_ge4sp.families.txt"
"$PYTHON" "$HERE/make_hogenom_full_families.py" \
    --out "$HERE/hogenom_full_ge4sp.families.txt"    # full: default manifest (12,408 -> 10,869)

# subsets to time: "NAME FAMFILE"
SUBSETS_DEFAULT=$'512 '"$HERE"$'/hogenom_512_ge4sp.families.txt\n1055 '"$HERE"$'/hogenom_1055_ge4sp.families.txt\nfull '"$HERE"$'/hogenom_full_ge4sp.families.txt'
SUBSETS="${SUBSETS:-$SUBSETS_DEFAULT}"

command -v nvidia-smi >/dev/null && \
  echo "GPU at start: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
_filter_tqdm() { grep --line-buffered -vE 'it/s|%\|'; }

echo "=== gpurec HOGENOM(>=4sp)-subset timing @ $(date -Is) | host=$(hostname) ==="
echo "    driver = $DRIVER"
SUMMARY="$HERE/hogenom_subsets_timing.summary.txt"; : > "$SUMMARY"

while IFS=' ' read -r NAME FAMFILE; do
  [ -z "${NAME:-}" ] && continue
  NFAM=$(grep -c '^- ' "$FAMFILE")
  WLOG="$HERE/hogenom_${NAME}.gpurec.warm.log"
  ULOG="$HERE/hogenom_${NAME}.gpurec.warmup.log"
  echo
  echo "################ subset=$NAME  (FAMFILE=$(basename "$FAMFILE"), F=$NFAM) ################"

  # ---- WARM-UP (CERT off; warms ~/.triton, time discarded) ----
  echo "  [warmup] $(date -Is) ..."
  DATASET=hogenom FAMFILE="$FAMFILE" CERT=0 \
    "$PYTHON" -u "$DRIVER" > "$ULOG" 2>&1 || { echo "  warmup FAILED (see $ULOG)"; exit 1; }

  # ---- WARM (measured; CERT on -> also gives convergence + NLL) ----
  echo "  [warm]   $(date -Is) ..."
  t0=$(date +%s)
  DATASET=hogenom FAMFILE="$FAMFILE" CERT=1 \
    "$PYTHON" -u "$DRIVER" 2>&1 | _filter_tqdm | tee "$WLOG" || { echo "  warm FAILED"; exit 1; }
  wall=$(( $(date +%s) - t0 ))

  line=$("$PYTHON" - "$WLOG" "$NAME" "$NFAM" "$wall" <<'PY'
import re, sys
log, name, nfam, wall = open(sys.argv[1]).read(), sys.argv[2], sys.argv[3], sys.argv[4]
def g(pat, d="?"):
    m = re.search(pat, log)
    return m.group(1) if m else d
fit  = g(r"optimize \(incl builds\+adam\)\s*=\s*([0-9.]+)s")
cert = g(r"final cert.*?=\s*([0-9.]+)s")
tot  = g(r"(?m)^\s*TOTAL\s*=\s*([0-9.]+)s")   # anchor: avoid matching 'REBATCH TOTAL'
conv = g(r"CONVERGED[^=]*=\s*([0-9]+/[0-9]+)")
nll  = g(r"total NLL\s*=\s*([0-9.]+)\s*bits")
print(f"{name:>5}  F={nfam:>6}  fit_s={fit:>7}  cert_s={cert:>6}  total_s={tot:>7}  wall_s={wall:>6}  "
      f"conv={conv}  NLL={nll} bits")
PY
)
  echo "  RESULT: $line"
  echo "$line" >> "$SUMMARY"
done <<< "$SUBSETS"

echo
echo "============== gpurec HOGENOM(>=4sp) genewise timing (WARM, traced recipe) =============="
echo "  fit_s = AleRax-comparable optimize time (no PD cert); wall_s = whole-process wall-clock"
cat "$SUMMARY"
echo "========================================================================================"
echo "logs: hogenom_<name>.gpurec.{warmup,warm}.log    family files: hogenom_<name>_ge4sp.families.txt"
echo "=== DONE @ $(date -Is) ==="
