#!/usr/bin/env python3
"""Combine the per-chunk AleRax outputs into the single head-to-head result.

Per-family genewise fits are independent, so the full-set result is exact: concatenate per-family
log-likelihoods (sum -> total NLL) and sum the per-chunk wall-clocks (sequential chunks -> total time).

    python combine_alerax_chunks.py CHUNKDIR TIMING.txt
"""
import glob, math, os, sys


def main():
    chunkdir, timingfile = sys.argv[1], sys.argv[2]
    # per-family log-likelihoods (AleRax per_fam_likelihoods.txt: "family <ll>", ll in nats, negative)
    rows, total_ll, nfam = [], 0.0, 0
    for f in sorted(glob.glob(os.path.join(chunkdir, "*.per_fam_likelihoods.txt"))):
        for ln in open(f):
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                ll = float(parts[-1])
            except ValueError:
                continue
            total_ll += ll
            nfam += 1
            rows.append(ln.rstrip("\n"))
    out = os.path.join(os.path.dirname(chunkdir.rstrip("/")), "alerax_hogenom_combined_likelihoods.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(rows) + ("\n" if rows else ""))

    total_s, chunk_lines = 0, []
    for ln in open(timingfile):
        p = ln.split()
        if len(p) == 3 and p[2].isdigit():
            total_s += int(p[2])
            chunk_lines.append((p[0], int(p[1]), int(p[2])))

    print("=" * 64)
    print("AleRax HOGENOM (>=4 species) — chunked, memory-capped, combined")
    print("=" * 64)
    for name, nf, sec in chunk_lines:
        print(f"  {name}: {nf:5d} families  {sec:6d}s")
    print("-" * 64)
    print(f"  TOTAL families      = {nfam}")
    print(f"  TOTAL wall-clock    = {total_s}s = {total_s/60:.1f} min  (sum of chunks)")
    print(f"  TOTAL log-likelihood= {total_ll:.1f} nats   ->  NLL = {-total_ll:.1f} nats")
    print(f"  (gpurec genewise full >=4: 758s fit; NLL 1,906,464 bits = {1906464*math.log(2):.0f} nats)")
    print(f"  combined per-family likelihoods -> {out}")


if __name__ == "__main__":
    sys.exit(main())
