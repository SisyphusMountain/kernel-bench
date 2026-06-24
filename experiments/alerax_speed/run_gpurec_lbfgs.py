"""§5.1 variant: post-Adam endgame via BATCHED L-BFGS-B (per-family, the optimizer AleRax uses)
instead of the FD-3x3 trust-region Newton. Front-end unchanged (clade_budget=900k, 5-step lr=1
clip-10 Adam). BatchedLBFGS keeps one L-BFGS history per family (row), strong-Wolfe line search,
box bounds [MIN_RATE,MAX_RATE] -> it IS L-BFGS-B, batched on GPU.

Structure: Adam warmup -> L-BFGS-B at pi16 over ALL families to |Pg|<TOL (no FD-Hessian, no per-iter
rebuild) -> escalate the still-unconverged stiff tail to pi64 L-BFGS-B -> final cold cert at
pi64/neu64 over all (|Pg| + 3x3 lam_min), same as the Newton recipe so the numbers are comparable.

Env: HIST=10 LS=strong_wolfe LBFGS16=60 LBFGS64=20 CHECK=2 ADAM=5 ADAM_LR=1.0 GRAD_CLIP=10
     CLADE_BUDGET=900000 PI=16 CERT_PI=64 NEU_OPT=16 NEU_CERT=64 TOL=1e-3 MIN_RATE=1e-6 MAX_RATE=2
"""
from __future__ import annotations
import os, sys, time, math, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions, BatchedLBFGS
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds

DEV = "cuda"; DT = torch.float32
DATASET = "archaea"
PI = int(os.environ.get("PI", "16")); CERT_PI = int(os.environ.get("CERT_PI", "64"))
NEU_OPT = int(os.environ.get("NEU_OPT", "16")); NEU_CERT = int(os.environ.get("NEU_CERT", "64"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = 1e-2
ADAM = int(os.environ.get("ADAM", "5")); ADAM_LR = float(os.environ.get("ADAM_LR", "1.0")); GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "10"))
CLADE_BUDGET = int(os.environ.get("CLADE_BUDGET", "900000"))
HIST = int(os.environ.get("HIST", "10")); LS = os.environ.get("LS", "strong_wolfe")
LBFGS16 = int(os.environ.get("LBFGS16", "60")); LBFGS64 = int(os.environ.get("LBFGS64", "20"))
CHECK = int(os.environ.get("CHECK", "2"))
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
FAMFILE = os.environ.get("FAMFILE", os.path.join(HERE, "archaea_ge4sp.families.txt"))
print(f"=== L-BFGS-B endgame  hist={HIST} ls={LS}  pi={PI}->cert{CERT_PI}  box[{MIN_RATE},{MAX_RATE}] ===", flush=True)


def sopts(pi, neu): return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": neu})
def build(paths, pi, neu):
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(pi, neu), clade_budget=CLADE_BUDGET)
    m.receiver_weights.requires_grad_(False); return m
def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); return lv.to(DT), g.to(DT)
def pgmax(th, g): return project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE).abs().amax(dim=1)
def clamp_(th): clamp_log_rate_(th, min_rate=MIN_RATE, max_rate=MAX_RATE); return th


fam_paths = [ln.split("=", 1)[1].strip() for ln in open(FAMFILE) if ln.startswith("starting_gene_tree")]
F_all = len(fam_paths)
print(f"[FAMFILE] {F_all} families", flush=True)

t0 = time.perf_counter()
m = build(fam_paths, PI, NEU_OPT)
torch.cuda.synchronize(); build_s = time.perf_counter() - t0
print(f"[build] F={F_all}  build={build_s:.1f}s", flush=True)

# ---- Adam warmup (the winning front-end) ----
ta = time.perf_counter()
lf = torch.zeros(F_all, 3, device=DEV, dtype=DT); clamp_(lf); lf.requires_grad_(True)
ad = torch.optim.Adam([lf], lr=ADAM_LR)
for _ in range(ADAM):
    _, g = lg(m, lf.detach()); lf.grad = g
    if GRAD_CLIP > 0: torch.nn.utils.clip_grad_norm_(lf, GRAD_CLIP)
    project_rate_gradient_(lf.detach(), lf.grad, min_rate=MIN_RATE, max_rate=MAX_RATE)
    ad.step()
    with torch.no_grad(): clamp_(lf)
sub = lf.detach().clone()
torch.cuda.synchronize(); adam_s = time.perf_counter() - ta
print(f"[adam] {ADAM} steps lr={ADAM_LR} clip={GRAD_CLIP}  dt={adam_s:.1f}s", flush=True)


def lbfgs_solve(m, sub0, max_it, tag, lbl):
    """Per-family L-BFGS-B until |Pg|<TOL or max_it iters. Returns (theta, n_steps, n_closure_evals)."""
    th = sub0.detach().clone().requires_grad_(True)
    opt = BatchedLBFGS([th], lr=1.0, history_size=HIST, max_iter=1, line_search_fn=LS,
                       lower_bound=float(TH_LO), upper_bound=float(TH_HI))
    nev = [0]
    def closure():
        lv, g = lg(m, th.detach()); th.grad = g; nev[0] += 1; return lv
    for it in range(max_it):
        opt.step(closure)
        if it % CHECK == 0 or it == max_it - 1:
            _, g = lg(m, th.detach()); pgm = pgmax(th.detach(), g); nev[0] += 1
            conv = pgm < TOL; frac = float(conv.float().mean())
            print(f"  [{tag} it{it:3d}] conv={frac*100:4.0f}%  |Pg|max={float(pgm.max()):.2e}  "
                  f"evals={nev[0]:4d}  t={time.perf_counter()-t0:.0f}s", flush=True)
            if bool(conv.all()): break
    return th.detach(), it + 1, nev[0]


# ---- pi16 bulk ----
t16 = time.perf_counter()
theta, n16, ev16 = lbfgs_solve(m, sub, LBFGS16, f"pi{PI}", "bulk")
del m; torch.cuda.empty_cache()
print(f"[pi{PI}] {n16} L-BFGS steps, {ev16} closure-evals, dt={time.perf_counter()-t16:.0f}s", flush=True)

# ---- escalate the still-unconverged stiff tail to pi64 ----
mc = build(fam_paths, CERT_PI, NEU_CERT)
_, g = lg(mc, theta); pg = pgmax(theta, g); unconv = pg >= TOL
n64 = ev64 = 0
if bool(unconv.any()):
    idx = torch.nonzero(unconv, as_tuple=False).squeeze(1)
    print(f"[escalate] {idx.numel()} families fail pi{PI} cert -> pi{CERT_PI} L-BFGS-B", flush=True)
    msub = build([fam_paths[j] for j in idx.tolist()], CERT_PI, NEU_OPT)
    th_sub, n64, ev64 = lbfgs_solve(msub, theta.index_select(0, idx), LBFGS64, f"pi{CERT_PI}", "tail")
    theta.index_copy_(0, idx, th_sub); del msub; torch.cuda.empty_cache()
opt_s = time.perf_counter() - t0

# ---- final cold cert at (CERT_PI, NEU_CERT) over ALL ----
tc = time.perf_counter()
_, g = lg(mc, theta); pg = pgmax(theta, g)
H = torch.zeros(F_all, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gp = lg(mc, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gm = lg(mc, tm)
    H[:, :, j] = (gp - gm) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
at_lo = (theta <= TH_LO + 1e-6); at_hi = (theta >= TH_HI - 1e-6); bound_active = (at_lo | at_hi).any(dim=1)
conv = pg < TOL; pd = lam_min > TOL
nll_bits = float(mc.genewise_loss_vector(theta=theta).sum()); nll_nats = nll_bits * math.log(2)
total = time.perf_counter() - t0; cert_s = time.perf_counter() - tc
print(f"\n{'='*64}\nL-BFGS-B ENDGAME  (archaea F={F_all}, hist={HIST} {LS})\n{'='*64}", flush=True)
print(f"  CONVERGED (|Pg|<{TOL}) = {int(conv.sum())}/{F_all}   |Pg|max={float(pg.max()):.2e}", flush=True)
print(f"  interior PD = {int((conv & pd & ~bound_active).sum())}   bound-active = {int(bound_active.sum())}   "
      f"unconverged = {int((~conv).sum())}", flush=True)
print(f"  total NLL = {nll_bits:.1f} bits = {nll_nats:.1f} nats   [AleRax: 226334 nats]", flush=True)
print(f"  L-BFGS steps: pi{PI}={n16} (+{ev16} evals)  pi{CERT_PI} tail={n64} (+{ev64} evals)", flush=True)
print(f"  build={build_s:.0f}s  adam={adam_s:.0f}s  optimize={opt_s:.0f}s  cert={cert_s:.0f}s  TOTAL={total:.0f}s", flush=True)
