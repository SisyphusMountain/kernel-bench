#!/usr/bin/env python3
"""Regenerate fig_speed.pdf: gpurec genewise fit wall-clock vs #families on Hogenom (>=4 species).

Data are the measured warm-run optimize times (gpurec.fit_genewise, single RTX 4090, traced recipe);
see Table~\\ref{tab:speed} and experiments/alerax_speed/run_gpurec_hogenom_subsets.sh.
    python make_fig_speed.py        # -> fig_speed.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (n_families [>=4 species], optimize wall-clock seconds) -- measured 2026-06-24
POINTS = [(506, 119), (1042, 219), (10869, 758)]
LABELS = ["Hogenom-512", "Hogenom-1055", "Hogenom-full"]
ALERAX = (10869, 17682)   # AleRax v1.4.0, 24 CPU cores (chunked + 80 GB-capped), same >=4 set
HERE = os.path.dirname(os.path.abspath(__file__))

ns = [n for n, _ in POINTS]
ts = [t for _, t in POINTS]
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.loglog(ns, ts, "o-", color="#1f5fae", lw=1.8, ms=7, label="gpurec (RTX 4090)", zorder=3)
# AleRax CPU baseline (same full >=4 set)
ax.loglog([ALERAX[0]], [ALERAX[1]], "s", color="#c0392b", ms=9, label="AleRax (24 CPU cores)", zorder=4)
ax.annotate("AleRax ($\\sim$23$\\times$ slower)", ALERAX,
            textcoords="offset points", xytext=(-6, 6), fontsize=7, ha="right", va="bottom", color="#c0392b")
# arrow marking the gpurec-full vs AleRax gap at the full set
ax.annotate("", xy=(10869, 758), xytext=(10869, 17682),
            arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.0), zorder=1)
for (n, t), lab in zip(POINTS, LABELS):
    ax.annotate(f"{lab}\n{n} fam", (n, t),
                textcoords="offset points", xytext=(8, -4), fontsize=7, va="top")
ax.set_xlabel("number of gene families ($\\geq 4$ species)")
ax.set_ylabel("wall-clock (s)")
ax.set_title("Genewise DTL-rate fitting: gpurec vs AleRax", fontsize=9)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
out = os.path.join(HERE, "fig_speed.pdf")
fig.savefig(out)
print(f"wrote {out}  (throughput {ns[0]/ts[0]:.1f} -> {ns[-1]/ts[-1]:.1f} fam/s)")
