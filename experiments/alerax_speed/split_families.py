#!/usr/bin/env python3
"""Split an AleRax `[FAMILIES]` file into K roughly-equal chunk files (deterministic, in order).

Per-family genewise rates are independent (AleRax `--model-parametrization PER-FAMILY` + fixed species
tree), so fitting each chunk separately gives bit-identical per-family rates and a summable likelihood.

    python split_families.py FAMILIES.txt K OUTDIR     # -> OUTDIR/chunk_00.families.txt ...
"""
import math, os, sys


def main():
    infile, K, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lines = [ln.rstrip("\n") for ln in open(infile)]
    fams = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("- "):
            gtree = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("starting_gene_tree") else None
            fams.append((lines[i], gtree))
            i += 2 if gtree else 1
        else:
            i += 1
    n = len(fams)
    per = math.ceil(n / K)
    os.makedirs(outdir, exist_ok=True)
    nc = 0
    for c in range(K):
        chunk = fams[c * per:(c + 1) * per]
        if not chunk:
            break
        with open(os.path.join(outdir, f"chunk_{c:02d}.families.txt"), "w") as fh:
            fh.write("[FAMILIES]\n")
            for name, gtree in chunk:
                fh.write(name + "\n")
                if gtree:
                    fh.write(gtree + "\n")
        nc += 1
        print(f"  chunk_{c:02d}: {len(chunk)} families")
    print(f"split {n} families -> {nc} chunks (<= {per} each)")


if __name__ == "__main__":
    sys.exit(main())
