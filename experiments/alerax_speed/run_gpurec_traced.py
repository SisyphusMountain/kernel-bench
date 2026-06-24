"""Thin benchmark wrapper over the library recipe ``gpurec.fit_genewise`` — times the genewise fit on a
dataset/famfile and prints a parser-compatible summary (optimize / final cert / TOTAL / CONVERGED / NLL).

The recipe itself now lives in the library (``gpurec/optim/genewise_fit.py``); this driver only resolves
the family set, maps env knobs to ``fit_genewise`` parameters, and reports timing. Run (warm cache):
  WT=<mapcv-merge worktree>
  FAMFILE=$PWD/hogenom_1055_ge4sp.families.txt PYTHONPATH=$WT:$WT/experiments/sanderson_cv \
  GPUREC_PREPROCESS_PATH=$WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so \
  DATASET=hogenom CERT=1 python -u run_gpurec_traced.py
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS
import gpurec

DATASET = os.environ.get("DATASET", "hogenom")
_FAM = os.environ.get("FAMILIES", "all"); N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
PIS = [int(x) for x in os.environ.get("PIS", os.environ.get("PI", "16,64")).split(",")]
NEU_OPT = int(os.environ.get("NEU_OPT", "16")); NEU_CERT = int(os.environ.get("NEU_CERT", "64"))
ADAM = int(os.environ.get("ADAM", "5")); ADAM_LR = float(os.environ.get("ADAM_LR", "1.0"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "10")); MAXIT = int(os.environ.get("MAXIT", "120"))
TOL = float(os.environ.get("TOL", "1e-3"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
_CB_DEFAULT = "900000" if DATASET == "archaea" else "80000"   # dataset-aware GPU occupancy default
CLADE_BUDGET = int(os.environ.get("CLADE_BUDGET", _CB_DEFAULT))
CERT = os.environ.get("CERT", "0") != "0"

_famfile = os.environ.get("FAMFILE")
if _famfile:
    paths = [ln.split("=", 1)[1].strip() for ln in open(_famfile) if ln.startswith("starting_gene_tree")]
    print(f"[FAMFILE] {len(paths)} families from {_famfile}", flush=True)
else:
    paths = DATASETS[DATASET]["families"](N_FAM)
F = len(paths)
print(f"=== gpurec.fit_genewise  {DATASET} F={F}  pi-tiers={PIS} neu_opt={NEU_OPT}(warm) neu_cert={NEU_CERT}  "
      f"cb={CLADE_BUDGET} cert={CERT} ===", flush=True)

t0 = time.perf_counter()
res = gpurec.fit_genewise(
    str(DATASETS[DATASET]["species_tree"]), paths,
    min_rate=MIN_RATE, max_rate=MAX_RATE, pi_tiers=PIS, neu_opt=NEU_OPT, neu_cert=NEU_CERT,
    clade_budget=CLADE_BUDGET, adam_steps=ADAM, adam_lr=ADAM_LR, grad_clip=GRAD_CLIP,
    tol=TOL, max_iter=MAXIT, certify=CERT, verbose=True,
)
total = time.perf_counter() - t0
fit = res["opt_seconds"]; cert_s = total - fit

print(f"\n{'='*60}\n  TIMING / CONVERGENCE\n{'='*60}", flush=True)
print(f"  optimize (incl builds+adam)= {fit:.0f}s", flush=True)
print(f"  final cert (cold pi64/neu64) = {cert_s:.0f}s", flush=True)
print(f"  TOTAL                      = {total:.0f}s", flush=True)
print(f"  newton steps = {res['n_steps']}   model (re)builds = {res['n_builds']}", flush=True)
if CERT:
    print(f"  CONVERGED (|Pg|<{TOL}) = {res['converged']}/{F}   |Pg|max={res['pg_max']:.2e}", flush=True)
    print(f"  interior PD = {res['interior_pd']}   bound-active = {res['bound_active']}   "
          f"unconverged = {res['unconverged']}   (of which dropped-prematurely = {res['premature_drops']})", flush=True)
    print(f"  total NLL = {res['loss_bits']:.1f} bits = {res['loss_nats']:.1f} nats", flush=True)
else:
    print(f"  (CERT=1 for the cold PD certificate + NLL over all families)", flush=True)
