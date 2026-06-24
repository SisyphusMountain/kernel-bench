#!/usr/bin/env python3
"""Genewise (per-family) DTL reconciliation with gpurec — a clean, dataset-agnostic CLI.

A thin wrapper over the library default recipe ``gpurec.fit_genewise`` (Adam warm-up -> box-constrained
trust-region Newton with convergence-based rebatching + pi-tier escalation + memory-gated warm-adjoint).
It fits per-family Duplication/Loss/Transfer rates by maximum likelihood — exactly the model AleRax runs
with ``--rec-model UndatedDTL --model-parametrization PER-FAMILY`` — on the SAME inputs:

  * species tree : a rooted newick.
  * gene trees   : one CCP `.ale` per family (as built for AleRax), OR a newick (distribution) per family.
                   Gene-tree leaves named `<Species>_<id>` map to the species by the prefix before the
                   first underscore (no mapping file — same convention as AleRax).

Output: a TSV `family<TAB>Dup<TAB>Loss<TAB>Trans` of per-family rates relative to speciation (= 2^theta),
directly comparable to AleRax's `model_parameters/<fam>_rates.txt`.

Usage (gpurec importable; Rust preprocess lib reachable via GPUREC_PREPROCESS_PATH):
    python reconcile.py --species-tree tree.newick --gene-trees 'ale_dir/*.ale' --out rates.tsv
"""
from __future__ import annotations
import argparse, os, sys, time

import torch
import gpurec
from gpurec.optim.genewise_fit import _resolve_gene_trees


def main():
    ap = argparse.ArgumentParser(description="Genewise DTL reconciliation with gpurec (AleRax-compatible inputs).")
    ap.add_argument("--species-tree", required=True, help="rooted newick species tree")
    ap.add_argument("--gene-trees", required=True, help="list of paths, a glob ('d/*.ale'), a directory, or a listfile")
    ap.add_argument("--out", default="rates.tsv", help="output TSV (family, Dup, Loss, Trans)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-rate", type=float, default=1e-6)
    ap.add_argument("--max-rate", type=float, default=10.0, help="upper box bound on each rate")
    ap.add_argument("--clade-budget", type=int, default=None, help="max clades per GPU batch (default: library default)")
    ap.add_argument("--certify", action="store_true", help="also run the final cold PD certificate (|Pg|, lam_min, NLL)")
    ap.add_argument("--fp64", action="store_true", help="double precision (slower, for tight tolerances)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-iteration progress")
    args = ap.parse_args()

    paths = _resolve_gene_trees(args.gene_trees)   # resolve once so output names align with the fitted rates
    print(f"[reconcile] {len(paths)} families  device={args.device} "
          f"dtype={'float64' if args.fp64 else 'float32'}", flush=True)
    t0 = time.perf_counter()
    res = gpurec.fit_genewise(
        args.species_tree, paths, device=args.device,
        dtype=torch.float64 if args.fp64 else torch.float32,
        min_rate=args.min_rate, max_rate=args.max_rate,
        clade_budget=args.clade_budget, certify=args.certify, verbose=not args.quiet,
    )

    rates = res["rates"].cpu()
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    with open(args.out, "w") as fh:
        fh.write("family\tDup\tLoss\tTrans\n")
        for name, (d, l, t) in zip(names, rates.tolist()):
            fh.write(f"{name}\t{d:.6g}\t{l:.6g}\t{t:.6g}\n")

    msg = f"[reconcile] done in {time.perf_counter()-t0:.0f}s  ({res['n_steps']} Newton steps)"
    if args.certify:
        msg += (f"  converged {res['converged']}/{res['n_families']}  "
                f"total NLL = {res['loss_bits']:.1f} bits = {res['loss_nats']:.1f} nats")
    print(msg, flush=True)
    print(f"[reconcile] per-family rates -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
