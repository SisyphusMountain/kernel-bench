#!/usr/bin/env bash
# Speed comparison: AleRax genewise (per-family UndatedDTL) on HOGENOM-full, restricted to families
# covering >= 4 species (10,869 of 12,408). CPU via MPI, 24 ranks. Timed end-to-end with /usr/bin/time.
#
# FAIRNESS with the gpurec genewise benchmark:
#   * same 666-leaf Hogenom-Core species tree (Morel2024) that run_cv/reconcile use;
#   * same per-family gene-tree distributions (ufboot1000.MFP.geneTree.newick, 1000 trees each).
#     AleRax reads ALL trees in each distribution file to build its CCP -- no flag controls this --
#     which matches gpurec's Rust preprocess (it also accumulates every tree). So the CCPs are built
#     from the identical 1000 trees on both sides.
#   * same >=4-covered-species subset, pre-filtered with the identical "species = prefix before first
#     '_'" rule AleRax uses by default, so no --min-covered-species flag is needed.
#   * --gene-tree-samples 0 : draw ZERO reconciled gene-tree samples from the inferred MAP distribution
#     (that is an OUTPUT feature unrelated to the CCP; we only want the optimized rates + likelihood,
#     so 0 avoids the extra sampling cost). Same value the archaea run used.
#   * --species-tree-search SKIP, --rate-optimizer LBFGSB, --model-parametrization PER-FAMILY,
#     --rec-model UndatedDTL : identical to the archaea AleRax run (run_alerax.sh) = genewise rate
#     fitting on a fixed species tree.
# Output (heavy: per-family ccps/rates) -> $OUT under /tmp; timing/stdout -> run_alerax_hogenom_full.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom
SPECIES_TREE="$ROOT/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"
FAMILIES="$HERE/hogenom_full_ge4sp.families.txt"
OUT="${OUT:-/tmp/alerax_hogenom_full_ge4sp_out}"
NP="${NP:-24}"

# (re)build the >=4-species families subset so the run is self-contained / reproducible
"${PYTHON:-python3}" "$HERE/make_hogenom_full_families.py"

rm -rf "$OUT"
NFAM=$(grep -c '^- ' "$FAMILIES")
# SAFETY: this single run loads ALL families' evaluators into RAM at once (>125 GB) and WILL exceed the
# cap -> it is OOM-killed cleanly by the cgroup (your DESKTOP is never the victim). For a run that
# COMPLETES, use ./run_alerax_hogenom_chunked.sh (splits into RAM-sized chunks; bit-exact per-family).
MEMMAX="${MEMMAX:-80G}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
echo "=== AleRax HOGENOM-full(>=4sp) run @ $(date -Is) | host=$(hostname) | np=$NP cap=$MEMMAX | ${NFAM} families ==="
echo "!! WARNING: the full set needs >125 GB; expect a clean OOM-kill at $MEMMAX. Use the chunked runner to finish."
systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 --quiet \
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
# keep the lightweight per-family likelihoods next to this script (heavy ccps stay in $OUT)
cp -f "$OUT/per_fam_likelihoods.txt" "$HERE/hogenom_full_per_fam_likelihoods.txt" 2>/dev/null || true
