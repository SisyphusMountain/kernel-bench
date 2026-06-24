#!/usr/bin/env bash
# MEMORY-SAFE AleRax genewise on HOGENOM (>=4 species, 10,869 families). AleRax loads ALL families'
# CCPs + evaluators into RAM at once (>125 GB) -> global OOM that kills the desktop. Two-part fix:
#   (1) every AleRax invocation runs inside an 80 GB cgroup memory cap (systemd-run --user --scope) so
#       if it ever exceeds the cap the cgroup OOM-killer kills ONLY AleRax -- the desktop is untouched;
#   (2) the family set is split into CHUNKS that fit in RAM. Per-family genewise rates are independent
#       (--model-parametrization PER-FAMILY + --species-tree-search SKIP), so chunking is BIT-EXACT:
#       combined per-family rates == single run, total NLL = sum of per-family log-likelihoods, total
#       wall-clock = sum of (sequential) chunk times. The FIRST chunk validates the size for the rest.
# Reproducible: rebuilds the deterministic >=4-species family file, fixed seed (AleRax default 123).
# Env: NP=24 CHUNKS=8 MEMMAX=80G PYTHON= OUTBASE=
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

ROOT=/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom
SPECIES_TREE="$ROOT/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"
PYTHON="${PYTHON:-/home/enzo/miniforge3/bin/python}"
NP="${NP:-24}"; CHUNKS="${CHUNKS:-8}"; MEMMAX="${MEMMAX:-80G}"
OUTBASE="${OUTBASE:-/tmp/alerax_hogenom_chunked_out}"
CHUNKDIR="$HERE/alerax_chunks"
TIMING="$HERE/alerax_chunked_timing.txt"

# 1. deterministic >=4-species family set (10,869)
"$PYTHON" "$HERE/make_hogenom_full_families.py"
FAMILIES="$HERE/hogenom_full_ge4sp.families.txt"
NFAM=$(grep -c '^- ' "$FAMILIES")

# 2. split into CHUNKS
rm -rf "$CHUNKDIR"; mkdir -p "$CHUNKDIR"
"$PYTHON" "$HERE/split_families.py" "$FAMILIES" "$CHUNKS" "$CHUNKDIR"

# 3. run each chunk inside the 80 GB memory cap, NP ranks
echo "=== AleRax chunked run @ $(date -Is) | host=$(hostname) | NP=$NP cap=$MEMMAX chunks=$CHUNKS | $NFAM families ==="
: > "$TIMING"
rm -rf "$OUTBASE"; mkdir -p "$OUTBASE"
for cf in "$CHUNKDIR"/chunk_*.families.txt; do
  name=$(basename "$cf" .families.txt)
  out="$OUTBASE/$name"; rm -rf "$out"
  nf=$(grep -c '^- ' "$cf")
  echo "--- $name: $nf families @ $(date -Is) | mem avail: $(free -g | awk '/Mem/{print $7}')G ---"
  t0=$(date +%s)
  if systemd-run --user --scope -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 --quiet \
       /usr/bin/time -v mpiexec -np "$NP" alerax \
         -f "$cf" -s "$SPECIES_TREE" -p "$out" \
         --rec-model UndatedDTL --model-parametrization PER-FAMILY \
         --species-tree-search SKIP --rate-optimizer LBFGSB --gene-tree-samples 0 \
       > "$CHUNKDIR/$name.log" 2>&1; then
    dt=$(( $(date +%s) - t0 ))
    cp -f "$out/per_fam_likelihoods.txt" "$CHUNKDIR/$name.per_fam_likelihoods.txt" 2>/dev/null || true
    echo "$name $nf $dt" >> "$TIMING"
    echo "    OK  wall=${dt}s  peakRSS=$(grep -i 'Maximum resident' "$CHUNKDIR/$name.log" | grep -oE '[0-9]+' | head -1) kB/rank"
  else
    rc=$?
    echo "    !! $name FAILED (rc=$rc). The 80G cap killed ONLY AleRax -- your desktop is safe."
    echo "       If it was OOM, re-run with more chunks:  CHUNKS=$((CHUNKS*2)) ./run_alerax_hogenom_chunked.sh"
    echo "       log: $CHUNKDIR/$name.log"
    echo "$name $nf FAILED" >> "$TIMING"
    exit 1
  fi
done

# 4. combine -> the head-to-head AleRax number (total wall-clock + total NLL)
"$PYTHON" "$HERE/combine_alerax_chunks.py" "$CHUNKDIR" "$TIMING"
echo "=== DONE @ $(date -Is) ==="
