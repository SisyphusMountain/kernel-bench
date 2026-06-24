#!/usr/bin/env python3
"""Build a >=4-covered-species AleRax-format families file for HOGENOM (genewise comparison).

A gene leaf is named `<Species>_<id>` (e.g. CLOCE_1.PE2286 -> species CLOCE), so the species is the
substring before the FIRST underscore -- the EXACT rule gpurec's Rust preprocess uses
(lib.rs: leaf_name.split_once('_')) AND AleRax's default gene<->species mapping. A family is kept iff
its representative tree covers >= MIN_SPECIES distinct species. The output is the same `[FAMILIES]`
format AleRax and gpurec's run_gpurec_traced.py (FAMFILE=) both read, so the identical family set is
fit by both tools.

No-arg invocation reproduces the FULL >=4-species set (10,869 of 12,408) used by the AleRax run:
    python make_hogenom_full_families.py                         # -> hogenom_full_ge4sp.families.txt

Subset builds (for the gpurec timing experiment):
    python make_hogenom_full_families.py --source-format namelist \\
        --source-file <worktree>/experiments/sanderson_cv/families_1055.txt \\
        --count 512 --out hogenom_512_ge4sp.families.txt
"""
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# default source = the canonical full-hogenom manifest (12,408 families, AleRax [FAMILIES] format)
DEFAULT_MANIFEST = ("/home/enzo/Documents/git/gpurec/gpurec/benchmarks/large_dataset_capacity/"
                    "generated/alerax_hogenom_core_all_families.txt")
DEFAULT_GT_TMPL = ("/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom/"
                   "families/{F}/gene_trees/ufboot1000.MFP.geneTree.newick")
_SPECIES = re.compile(r"[(,]([A-Za-z][A-Za-z0-9]*)_")   # leaf species = token before first underscore


def read_family_names(path, fmt):
    """`manifest` = AleRax [FAMILIES] file (lines `- CLU_...`); `namelist` = one family name per line."""
    if fmt == "manifest":
        return [ln[2:].strip() for ln in open(path) if ln.startswith("- ")]
    return [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")]


def covered_species(gt_path):
    with open(gt_path) as fh:
        first_tree = fh.readline()        # representative tree; its leaf set == family coverage
    return set(_SPECIES.findall(first_tree))


def main():
    ap = argparse.ArgumentParser(description="Build a >=N-species AleRax/gpurec families file for HOGENOM.")
    ap.add_argument("--source-file", default=DEFAULT_MANIFEST, help="family-name source")
    ap.add_argument("--source-format", choices=["manifest", "namelist"], default="manifest")
    ap.add_argument("--count", default="all", help="keep the first N source families before filtering, or 'all'")
    ap.add_argument("--gt-tmpl", default=DEFAULT_GT_TMPL, help="gene-tree path template with {F}")
    ap.add_argument("--min-species", type=int, default=int(os.environ.get("MIN_SPECIES", "4")))
    ap.add_argument("--out", default=os.path.join(HERE, "hogenom_full_ge4sp.families.txt"))
    a = ap.parse_args()

    names = read_family_names(a.source_file, a.source_format)
    if a.count not in ("all", "0", ""):
        names = names[: int(a.count)]
    kept, dropped, missing = [], 0, 0
    for n in names:
        p = a.gt_tmpl.format(F=n)
        if not os.path.exists(p):
            missing += 1
            continue
        if len(covered_species(p)) >= a.min_species:
            kept.append((n, p))
        else:
            dropped += 1
    with open(a.out, "w") as o:
        o.write("[FAMILIES]\n")
        for n, p in kept:
            o.write(f"- {n}\nstarting_gene_tree = {p}\n")
    print(f"source={os.path.basename(a.source_file)} count={a.count}  considered={len(names)}  "
          f"missing={missing}  kept(>= {a.min_species} sp)={len(kept)}  dropped(< {a.min_species})={dropped}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
