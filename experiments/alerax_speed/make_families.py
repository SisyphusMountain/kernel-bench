#!/usr/bin/env python3
"""Build the AleRax families file for the §5.1 speed comparison (archaea, genewise).

Includes only families covering >= MIN_SPECIES distinct species (default 4), computed
from each .ale's representative gene tree leaf set (gene leaf = `<Species>_<id>`, so the
species is the prefix before the first underscore). Families with strictly fewer than
MIN_SPECIES covered species are dropped, so AleRax needs no --min-covered-species flag.

The same family set must be used for the gpurec genewise benchmark for a fair comparison.

Usage:
    python make_families.py            # writes archaea_ge4sp.families.txt next to this script
"""
import glob, os, re, sys

ALEDIR = os.environ.get(
    "ALEDIR",
    "/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_archaea_davin2017/"
    "ale_gene_tree_distributions/main_families_ge4seq",
)
MIN_SPECIES = int(os.environ.get("MIN_SPECIES", "4"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "archaea_ge4sp.families.txt"))

_LEAF = re.compile(r"[(,]([A-Za-z][A-Za-z0-9]*)_[0-9]+:")


def covered_species(ale_path):
    with open(ale_path) as fh:
        fh.readline()              # "#constructor_string"
        tree = fh.readline()       # representative gene tree; leaf set == family coverage
    return set(_LEAF.findall(tree))


def main():
    files = sorted(glob.glob(os.path.join(ALEDIR, "*.ale")))
    kept, dropped = [], 0
    for f in files:
        if len(covered_species(f)) >= MIN_SPECIES:
            kept.append(f)
        else:
            dropped += 1
    with open(OUT, "w") as o:
        o.write("[FAMILIES]\n")
        for f in kept:
            o.write(f"- {os.path.basename(f)[:-4]}\nstarting_gene_tree = {f}\n")
    print(f"total={len(files)}  kept(>= {MIN_SPECIES} species)={len(kept)}  dropped(< {MIN_SPECIES})={dropped}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
