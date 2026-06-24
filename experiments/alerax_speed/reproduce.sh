#!/usr/bin/env bash
# Single entry point to reproduce the gpurec genewise throughput on HOGENOM (>=4 species, 3 sizes).
# This is the GPU side and takes minutes. The AleRax CPU baseline (hours) is launched separately --
# see the printed command and README.md. Override machine paths via env: GPUREC_WT, PYTHON, PREPROCESS_SO.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "############################################################################"
echo "# gpurec genewise throughput  (HOGENOM >=4 species: 512/1055/full subsets) #"
echo "#   recipe = the library default gpurec.fit_genewise (warmup + warm timing) #"
echo "############################################################################"
"$HERE/run_gpurec_hogenom_subsets.sh"

echo
echo "=> gpurec results: $HERE/hogenom_subsets_timing.summary.txt"
echo
echo "############################################################################"
echo "# AleRax CPU baseline (>=4 species, full 10,869 families) -- LONG (hours).  #"
echo "# Run it DETACHED so it survives the shell, then compare against the table: #"
echo "#                                                                          #"
echo "#   nohup NP=24 $HERE/run_alerax_hogenom_full.sh \\"
echo "#       > $HERE/run_alerax_hogenom_full.log 2>&1 &                          #"
echo "############################################################################"
