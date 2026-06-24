"""PROFILING driver: FD-Newton+rebatch recipe (cb=900k, 5-step lr=1/clip=10 Adam) instrumented to
capture, for EVERY iteration, the wall-time AND the per-family convergence state, so we can study:
  (1) COARSE: the per-step time distribution across the whole run (not averaged; rebatching makes it
      very non-stationary).
  (2) FINE:  how many families in the active batch are ALREADY converged (|Pg|<TOL) yet still stepped
      -> wasted family-steps (the cost of lingering between drop checks).
Dumps a flat per-iteration record list to OUT_JSON (default /tmp/profile_steps.json). Each iteration is
timed with a host sync (adds a little overhead vs the 230s production run -- fine for distributions).
Env mirrors run_gpurec_traced.py; defaults set to the winning recipe.
"""
from __future__ import annotations
import os, sys, time, json, math
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds
from gpurec.core.inference.solver import solve_forward_residual

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "archaea")
PIS = [int(x) for x in os.environ.get("PIS", "16,64").split(",")]
CERT_PI = int(os.environ.get("CERT_PI", str(max(PIS))))
NEU_OPT = int(os.environ.get("NEU_OPT", "16")); NEU_CERT = int(os.environ.get("NEU_CERT", "64"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = 1e-2; MU = 1e-2; TRUST = 2.0
CHECK = int(os.environ.get("CHECK", "4")); FRAC = float(os.environ.get("FRAC", "0.30"))
WARM = os.environ.get("WARM", "1") != "0"
VERIFY_DROP = os.environ.get("VERIFY_DROP", "1") != "0"
ADAM = int(os.environ.get("ADAM", "5")); ADAM_LR = float(os.environ.get("ADAM_LR", "1.0")); GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "10"))
MAXIT = int(os.environ.get("MAXIT", "120")); HESS_EVERY = int(os.environ.get("HESS_EVERY", "5"))
CLADE_BUDGET = int(os.environ.get("CLADE_BUDGET", "900000"))
FWD_TOL = float(os.environ.get("FWD_TOL", "1e-3"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/profile_steps.json")
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
FAMFILE = os.environ.get("FAMFILE", os.path.join(HERE, "archaea_ge4sp.families.txt"))
print(f"=== PROFILE FD-Newton+rebatch  pis={PIS} adam={ADAM}/lr{ADAM_LR}/clip{GRAD_CLIP} cb={CLADE_BUDGET} ===", flush=True)

EV = []   # flat per-event record list
def sopts(pi, neu): return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": neu})
def build(paths, pi, neu, tag):
    torch.cuda.synchronize(); t = time.perf_counter()
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(pi, neu), clade_budget=CLADE_BUDGET)
    m.receiver_weights.requires_grad_(False)
    torch.cuda.synchronize(); EV.append(dict(phase="build", tag=tag, n=len(paths), pi=pi, neu=neu, dt_ms=(time.perf_counter()-t)*1000))
    return m
def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); return lv.to(DT), g.to(DT)
def pgvec(th, g): return project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE).abs().amax(dim=1)
def clamp_(th): clamp_log_rate_(th, min_rate=MIN_RATE, max_rate=MAX_RATE); return th
def set_warm(on): os.environ["GPUREC_WARM_ADJOINT"] = "1" if on else os.environ.pop("GPUREC_WARM_ADJOINT", "0") and "0"

t0 = time.perf_counter()
fam_paths = [ln.split("=", 1)[1].strip() for ln in open(FAMFILE) if ln.startswith("starting_gene_tree")]
F_all = len(fam_paths)
print(f"[FAMFILE] {F_all} families", flush=True)
theta = torch.zeros(F_all, 3, device=DEV, dtype=DT); clamp_(theta)
active = torch.arange(F_all, device=DEV)
gstep = 0   # global newton-step counter

def forward_resid(m, th, pi):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = torch.zeros(len(m.families), device=DEV, dtype=torch.float32); rw = m.receiver_weights.detach()
    with torch.no_grad():
        for static in m.batch_statics:
            r = solve_forward_residual(static, m._theta_for_static(static, th), rw, pi_iters=pi)
            out[static.family_index_tensor.to(DEV)] = r.to(DEV)
    torch.cuda.synchronize(); EV.append(dict(phase="resid", n=int(len(m.families)), dt_ms=(time.perf_counter()-t)*1000))
    return out

if WARM: os.environ["GPUREC_WARM_ADJOINT"] = "1"
carry = None
for pi_idx, PI_CUR in enumerate(PIS):
    if carry is not None: active = carry
    if active.numel() == 0: break
    last_tier = pi_idx == len(PIS) - 1
    carry = active[:0].clone()
    m = build([fam_paths[j] for j in active.tolist()], PI_CUR, NEU_OPT, tag=f"pi{PI_CUR}")
    sub = theta.index_select(0, active).clone()
    if pi_idx == 0:                                        # Adam warmup (winning front-end)
        torch.cuda.synchronize(); ta = time.perf_counter()
        lf = sub.clone().requires_grad_(True); ad = torch.optim.Adam([lf], lr=ADAM_LR)
        for _ in range(ADAM):
            _, g = lg(m, lf.detach()); lf.grad = g
            if GRAD_CLIP > 0: torch.nn.utils.clip_grad_norm_(lf, GRAD_CLIP)
            project_rate_gradient_(lf.detach(), lf.grad, min_rate=MIN_RATE, max_rate=MAX_RATE)
            ad.step()
            with torch.no_grad(): clamp_(lf)
        sub = lf.detach().clone()
        torch.cuda.synchronize(); EV.append(dict(phase="adam", n=int(active.numel()), steps=ADAM, dt_ms=(time.perf_counter()-ta)*1000))
    Hd = None
    for it in range(MAXIT):
        if active.numel() == 0: break
        torch.cuda.synchronize(); _ts = time.perf_counter()
        lv, g = lg(m, sub)
        pg = pgvec(sub, g); nconv = int((pg < TOL).sum()); pgmx = float(pg.max())
        rec = dict(phase="newton", gstep=gstep, it=it, pi=PI_CUR, n_active=int(active.numel()),
                   n_conv=nconv, pgmax=pgmx, hess=0, drop=0, certverify=0, rebuilt=0)
        did_continue = False
        if it % CHECK == 0:
            conv = pg < TOL; frac = float(conv.float().mean())
            if frac > FRAC and bool(conv.any()):
                cert_ok = conv.clone()
                if VERIFY_DROP:
                    rec["certverify"] = 1
                    _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
                    m.solver_options = sopts(CERT_PI, NEU_CERT)
                    cert_ok = conv & (pgvec(sub, lg(m, sub)[1]) < TOL)
                    m.solver_options = sopts(PI_CUR, NEU_OPT)
                    if _w: os.environ["GPUREC_WARM_ADJOINT"] = _w
                drop = cert_ok; defer = torch.zeros_like(conv); reject = conv & ~cert_ok
                if not last_tier and bool(reject.any()):
                    resid = forward_resid(m, sub, PI_CUR); defer = reject & (resid > FWD_TOL)
                if bool(drop.any()) or bool(defer.any()):
                    rec["drop"] = int(drop.sum())
                    if bool(drop.any()):
                        theta.index_copy_(0, active[drop], sub[drop])
                    if bool(defer.any()):
                        theta.index_copy_(0, active[defer], sub[defer]); carry = torch.cat([carry, active[defer]])
                    keep = ~(drop | defer); active = active[keep]; sub = sub[keep].clone()
                    if active.numel() == 0:
                        torch.cuda.synchronize(); rec["dt_ms"] = (time.perf_counter()-_ts)*1000; EV.append(rec); break
                    torch.cuda.synchronize(); t = time.perf_counter()
                    del m; torch.cuda.empty_cache()
                    set_warm(True); m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]),
                        [str(fam_paths[j]) for j in active.tolist()], mode="genewise", device=DEV,
                        solver_options=sopts(PI_CUR, NEU_OPT), clade_budget=CLADE_BUDGET)
                    m.receiver_weights.requires_grad_(False)
                    torch.cuda.synchronize(); rec["rebuilt"] = 1
                    EV.append(dict(phase="rebuild", n=int(active.numel()), pi=PI_CUR, dt_ms=(time.perf_counter()-t)*1000))
                    Hd = None; did_continue = True
        if not did_continue:
            if it % HESS_EVERY == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                rec["hess"] = 1
                H = torch.zeros(sub.shape[0], 3, 3, device=DEV, dtype=DT)
                for j in range(3):
                    tp = sub.clone(); tp[:, j] += FD_EPS; _, gp = lg(m, tp)
                    tm = sub.clone(); tm[:, j] -= FD_EPS; _, gm = lg(m, tm)
                    H[:, :, j] = (gp - gm) / (2 * FD_EPS)
                H = 0.5 * (H + H.transpose(1, 2)); e, V = torch.linalg.eigh(H)
                Hd = V @ torch.diag_embed(e.clamp(min=MU)) @ V.transpose(1, 2)
            fixed = ((sub >= TH_HI - 1e-6) & (g < 0)) | ((sub <= TH_LO + 1e-6) & (g > 0)); free = (~fixed).float()
            Hred = Hd * free.unsqueeze(1) * free.unsqueeze(2) + torch.diag_embed(1.0 - free)
            delta = -torch.linalg.solve(Hred, (g * free).unsqueeze(-1)).squeeze(-1)
            dn = delta.norm(dim=1, keepdim=True); sub = clamp_(sub + delta * (TRUST / dn.clamp(min=TRUST)))
            gstep += 1
        torch.cuda.synchronize(); rec["dt_ms"] = (time.perf_counter()-_ts)*1000; EV.append(rec)
    if active.numel() > 0:
        theta.index_copy_(0, active, sub); carry = torch.cat([carry, active])
    del m; torch.cuda.empty_cache()
os.environ.pop("GPUREC_WARM_ADJOINT", None)
opt_s = time.perf_counter() - t0

# ---- cert (timed, not per-step) ----
torch.cuda.synchronize(); tc = time.perf_counter()
mfull = build(fam_paths, CERT_PI, NEU_CERT, tag="cert")
_, g = lg(mfull, theta); pg = pgvec(theta, g)
nll = float(mfull.genewise_loss_vector(theta=theta).sum())
torch.cuda.synchronize(); cert_s = time.perf_counter() - tc; total = time.perf_counter() - t0
EV.append(dict(phase="cert", n=F_all, dt_ms=cert_s*1000))
json.dump(dict(meta=dict(F=F_all, pis=PIS, opt_s=opt_s, cert_s=cert_s, total_s=total,
              nll_nats=nll*math.log(2), conv=int((pg < TOL).sum()), pgmax=float(pg.max())), events=EV),
          open(OUT_JSON, "w"))
print(f"[done] opt={opt_s:.0f}s cert={cert_s:.0f}s total={total:.0f}s  conv={int((pg<TOL).sum())}/{F_all} "
      f"|Pg|max={float(pg.max()):.2e}  nll={nll*math.log(2):.0f}nats  -> {OUT_JSON}  ({len(EV)} events)", flush=True)
