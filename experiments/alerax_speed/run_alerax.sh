#!/usr/bin/env bash
# §5.1 speed comparison: AleRax genewise (per-family UndatedDTL) on the archaea
# >=4-species family subset, fixed species tree, on CPU via MPI. Timed.
#
# Reproduces against the SAME inputs gpurec genewise uses (same species tree, same
# .ale CCPs). Output (heavy: per-family ccps/rates) goes to $OUT under /tmp; the
# timing + stdout are tee'd to run.log next to this script (kept under git).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_archaea_davin2017
SPECIES_TREE="$ROOT/species_reference/reference_species_tree.newick"
FAMILIES="$HERE/archaea_ge4sp.families.txt"
OUT="${OUT:-/tmp/alerax_archaea_out}"
# default to all PHYSICAL cores (Open MPI's slot count); hyperthreads are not counted
# as slots. This machine: Intel i9-13900K = 24 physical cores (8 P + 16 E), 32 threads.
NP="${NP:-$(lscpu -p=core | grep -v '^#' | sort -nu | wc -l)}"

# (re)build the families file so the run is self-contained / reproducible
"${PYTHON:-python3}" "$HERE/make_families.py"

rm -rf "$OUT"
echo "=== AleRax run @ $(date -Is) | host=$(hostname) | np=$NP ==="
/usr/bin/time -v mpiexec -np "$NP" alerax \
  -f "$FAMILIES" \
  -s "$SPECIES_TREE" \
  -p "$OUT" \
  --rec-model UndatedDTL \
  --model-parametrization PER-FAMILY \
  --species-tree-search SKIP \
  --rate-optimizer LBFGSB \
  --gene-tree-samples 0
echo "=== DONE @ $(date -Is) ==="
# keep the lightweight results next to this script (the heavy ccps stay in $OUT)
cp -f "$OUT/per_fam_likelihoods.txt" "$HERE/per_fam_likelihoods.txt" 2>/dev/null || true
