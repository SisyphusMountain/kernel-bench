"""Genewise convergence by WARM-START + CONVERGENCE-BASED rebatching.

One fixed (pi, neumann=NEU_OPT) config with adjoint WARM-START (recovers NEU_CERT-quality gradients at
NEU_OPT cost). Optimize ALL families with bounded active-set trust-region Newton. Every CHECK iters, detect
families whose LIKELIHOOD has plateaued (|loss(it) - loss(it-CHECK)| < CONV_TOL); once more than FRAC of the
ACTIVE batch has plateaued, FREEZE those families (park their theta) and DROP them -> rebuild the model over
only the survivors. The per-step cost shrinks as families finish, so the long tail of hard families runs on a
small batch. Final certificate at NEU_CERT COLD over ALL families (authoritative |Pg| + 3x3 lam_min); this
also flags any family that was dropped PREMATURELY (loss plateaued but |Pg| still > tol).

Why this beats pi-tier escalation here: warm-start already gives the stiff families an accurate gradient at
neu16, so the only adaptivity left worth paying for is shrinking the batch as families converge.

DEFAULT recipe = PIS=16,64 + REBATCH_RESID=1: run the bulk at pi=16, escalate ONLY the forward-stiff families
(forward residual > FWD_TOL) to pi=64. Full hogenom 12408: 1507s vs 2003s pi64-single, identical convergence.
Set REBATCH_RESID=0 (or PIS=64) to recover the single-tier convergence-drop recipe.

Env: DATASET=hogenom|hogenom_full|archaea FAMILIES=all|N PIS=16,64 REBATCH_RESID=1 FWD_TOL=1e-3 NEU_OPT=16
     NEU_CERT=64 MIN_RATE=1e-6 MAX_RATE=2 TOL=1e-3 CHECK=4 FRAC=0.30 ADAM=20 MAXIT=160 CLADE_BUDGET=80000 OUT_JSON=

  WT=$(git rev-parse --show-toplevel)
  GPUREC_PREPROCESS_PATH=$WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so PYTHONPATH=$WT \
  DATASET=hogenom FAMILIES=all python -u experiments/sanderson_cv/bench_genewise_warm_rebatch.py
"""
from __future__ import annotations
import os, sys, time, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds
from gpurec.core.inference.solver import solve_forward_residual

DEV = "cuda"; DT = torch.float32
DATASET = os.environ.get("DATASET", "hogenom")
_FAM = os.environ.get("FAMILIES", "all"); N_FAM = None if _FAM in ("all", "0", "") else int(_FAM)
# pi-ESCALATION: run convergence-drop at PIS[0]; families that STALL (stiff -- need higher pi to converge)
# escalate to PIS[1], etc. Drops + final cert are verified at CERT_PI (the authoritative high pi) so a family
# is never frozen on a low-pi-truncated optimum. PIS defaults to the single PI (backward compatible).
PIS = [int(x) for x in os.environ.get("PIS", os.environ.get("PI", "16,64")).split(",")]
CERT_PI = int(os.environ.get("CERT_PI", str(max(PIS))))
ESC_PATIENCE = int(os.environ.get("ESC_PATIENCE", "3"))   # checks with no drop AFTER a drop -> escalate
STUCK_MAX = int(os.environ.get("STUCK_MAX", "10"))        # checks with no drop at all -> escalate (fallback)
NEU_OPT = int(os.environ.get("NEU_OPT", "16")); NEU_CERT = int(os.environ.get("NEU_CERT", "64"))
MIN_RATE = float(os.environ.get("MIN_RATE", "1e-6")); MAX_RATE = float(os.environ.get("MAX_RATE", "2"))
TOL = float(os.environ.get("TOL", "1e-3")); FD_EPS = 1e-2; MU = 1e-2; TRUST = 2.0
CONV_TOL = float(os.environ.get("CONV_TOL", "1e-2"))   # per-family loss-plateau threshold (DROP_BY=loss only)
CHECK = int(os.environ.get("CHECK", "4")); FRAC = float(os.environ.get("FRAC", "0.30"))
# A family is "converged" (droppable) by its projected GRADIENT |Pg|<TOL (DEFAULT, reliable & free -- g is
# already computed each Newton step) or by LIKELIHOOD plateau |loss(it)-loss(it-CHECK)|<CONV_TOL. Measured:
# loss-plateau drops ~22% of hogenom families PREMATURELY (flat-loss region != minimum); grad does not.
DROP_BY = os.environ.get("DROP_BY", "grad")
# Warm-start cache is ~active_clades*S*4 bytes; on a big initial batch that can exceed GPU memory (full
# hogenom: 12408 fam -> 19.5GB). Gate it: warm ON only once the active batch <= WARM_MAX_FAM. The big
# early batch runs cold (its easy families are NOT stiff, and cold |Pg| is conservatively high so they
# never drop prematurely); warm engages for the small stiff tail, where it matters and the cache is small.
WARM_MAX_FAM = int(os.environ.get("WARM_MAX_FAM", "1000000000"))
# The opt |Pg| uses NEU_OPT (cold neu16 on the big batch), which can be biased BELOW tol -> premature drops
# (measured: 383/12408 on full hogenom). VERIFY_DROP re-checks the converged subset at NEU_CERT cold before
# freezing, so a family is dropped only if it is genuinely converged by the authoritative backward.
VERIFY_DROP = os.environ.get("VERIFY_DROP", "1") != "0"
ADAM = int(os.environ.get("ADAM", "20")); MAXIT = int(os.environ.get("MAXIT", "120")); HESS_EVERY = int(os.environ.get("HESS_EVERY", "5"))
CLADE_BUDGET = int(os.environ.get("CLADE_BUDGET", "80000"))
# REBATCH BY RESIDUAL (opt-in). The pi-tier escalation (STUCK_MAX/ESC_PATIENCE) escalates the WHOLE stalled
# batch to the next pi -- but most stalled families are not truncation-stiff, they just need more Newton steps,
# so bumping their pi (and clearing their warm cache) is wasted work (this is why PIS=16,64 churned 750 families
# and lost to single-tier). Instead: measure each family's FORWARD fixed-point residual at the current pi, and
# escalate ONLY the families that are both (a) forward-stiff (resid > FWD_TOL => the pi=16 gradient is
# truncation-biased so |Pg| can never reach tol) and (b) stuck (|Pg| stopped dropping). Non-stiff slow families
# keep converging at the low pi. Eager: defers the moment the residual flags a family, not at a global counter.
REBATCH_RESID = os.environ.get("REBATCH_RESID", "1") != "0"   # DEFAULT ON: the fastest full-hogenom recipe
FWD_TOL = float(os.environ.get("FWD_TOL", "1e-3"))           # forward last-update > this => under-converged at pi
IMPROVE_FRAC = float(os.environ.get("IMPROVE_FRAC", "0.8"))  # |Pg|(it) > IMPROVE_FRAC*|Pg|(it-CHECK) => stuck
TH_LO, TH_HI = log2_rate_bounds(MIN_RATE, MAX_RATE)
print(f"=== genewise WARM + CONVERGENCE-REBATCH {DATASET} fam={_FAM}  pi-tiers={PIS} cert_pi={CERT_PI} "
      f"neu_opt={NEU_OPT}(warm) neu_cert={NEU_CERT}  drop |Pg|<{TOL} when >{FRAC*100:.0f}% (verify={VERIFY_DROP}) ===", flush=True)


def sopts(pi, neu): return SolverOptions(**{**_CV_SO, "pi_iters": pi, "neumann_terms": neu})
def build(paths, pi, neu):
    m = GeneReconModel(str(DATASETS[DATASET]["species_tree"]), [str(x) for x in paths],
                       mode="genewise", device=DEV, solver_options=sopts(pi, neu), clade_budget=CLADE_BUDGET)
    m.receiver_weights.requires_grad_(False); return m
def lg(m, th):
    lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True); return lv.to(DT), g.to(DT)
def pgmax(th, g): return project_rate_gradient_(th, g.clone(), min_rate=MIN_RATE, max_rate=MAX_RATE).abs().amax(dim=1)
def clamp_(th): clamp_log_rate_(th, min_rate=MIN_RATE, max_rate=MAX_RATE); return th
def set_warm(n):                                       # enable warm-start only when the active batch fits in memory
    if n <= WARM_MAX_FAM: os.environ["GPUREC_WARM_ADJOINT"] = "1"
    else: os.environ.pop("GPUREC_WARM_ADJOINT", None)


t0 = time.perf_counter()
# FAMFILE override (for the §5.1 AleRax head-to-head): use the EXACT same family set
# AleRax ran on (the >=4-species subset from make_families.py), read from its families file.
_famfile = os.environ.get("FAMFILE")
if _famfile:
    fam_paths = [ln.split("=", 1)[1].strip() for ln in open(_famfile) if ln.startswith("starting_gene_tree")]
    print(f"[FAMFILE] {len(fam_paths)} families from {_famfile}", flush=True)
else:
    fam_paths = DATASETS[DATASET]["families"](N_FAM)
F_all = len(fam_paths)
theta = torch.zeros(F_all, 3, device=DEV, dtype=DT); clamp_(theta)
active = torch.arange(F_all, device=DEV)
was_dropped = torch.zeros(F_all, dtype=torch.bool, device=DEV)   # dropped mid-run (vs optimized to the end)

rebatch_log = []; defer_log = []
def build_active(pi):                                            # build over the current `active` at pi (warm-gated)
    set_warm(active.numel())
    return build([fam_paths[j] for j in active.tolist()], pi, NEU_OPT)
def forward_resid(m, th, pi):                                   # per-family forward last-update residual at pi
    out = torch.zeros(len(m.families), device=DEV, dtype=torch.float32)  # |pi_k - pi_{k-1}| max over each fam's clades
    rw = m.receiver_weights.detach()
    with torch.no_grad():
        for static in m.batch_statics:
            r = solve_forward_residual(static, m._theta_for_static(static, th), rw, pi_iters=pi)
            out[static.family_index_tensor.to(DEV)] = r.to(DEV)
    return out

carry = None                                                    # families deferred to the NEXT pi tier (REBATCH_RESID)
for pi_idx, PI_CUR in enumerate(PIS):                            # ascending pi tiers; stiff residual escalates
    if REBATCH_RESID and carry is not None: active = carry      # this tier works only the deferred-stiff carry-over
    if active.numel() == 0: break
    last_tier = pi_idx == len(PIS) - 1
    carry = active[:0].clone()                                  # reset: deferred families accumulate here this tier
    m = build_active(PI_CUR)
    sub = theta.index_select(0, active).clone()
    if pi_idx == 0:                                             # Adam warmup once, on the full batch
        lf = sub.clone().requires_grad_(True); ad = torch.optim.Adam([lf], lr=0.05)
        for _ in range(ADAM):
            _, g = lg(m, lf.detach()); lf.grad = g; ad.step()
            with torch.no_grad(): clamp_(lf)
        sub = lf.detach().clone()
    Hd = None; loss_ref = None; since_drop = 0; had_drop = False; pg_prev = None
    for it in range(MAXIT):
        if active.numel() == 0: break
        lv, g = lg(m, sub)
        if it % CHECK == 0:                                     # convergence-based drop check
            pgm = pgmax(sub, g)
            if DROP_BY == "loss":
                conv = ((loss_ref - lv).abs() < CONV_TOL) if (loss_ref is not None and loss_ref.shape[0] == lv.shape[0]) \
                    else torch.zeros_like(pgm, dtype=torch.bool)
            else:                                              # projected gradient |Pg|<TOL (reliable, free)
                conv = pgm < TOL
            frac = float(conv.float().mean())
            print(f"  [pi{PI_CUR:3d} it{it:3d}] active={active.numel():5d} conv={frac*100:4.0f}% "
                  f"|Pg|max={float(pgm.max()):.2e} t={time.perf_counter()-t0:.0f}s", flush=True)
            if REBATCH_RESID:
                # At a drop event (frac>FRAC), split the cheap-converged families with the cert + forward residual:
                #   DROP   = cheap-converged AND cert-converged                      -> truly at the minimum, freeze
                #   ESCALATE = cheap-converged but FAILS cert AND forward-stiff      -> the pi gradient is truncation-
                #              (resid>FWD_TOL)                                           biased, can't certify here ->
                #                                                                        defer to the next (higher) pi.
                # Both are decided at the SAME event so escalation is batched (no one-family-per-check trickle), and
                # forward-stiff families leave immediately instead of churning the low-pi tier to MAXIT.
                if frac > FRAC and bool(conv.any()):
                    cert_ok = conv.clone()
                    if VERIFY_DROP:                            # authoritative |Pg| at (CERT_PI, NEU_CERT) cold
                        _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
                        m.solver_options = sopts(CERT_PI, NEU_CERT)
                        cert_ok = conv & (pgmax(sub, lg(m, sub)[1]) < TOL)
                        m.solver_options = sopts(PI_CUR, NEU_OPT)
                        if _w: os.environ["GPUREC_WARM_ADJOINT"] = _w
                    drop = cert_ok
                    defer = torch.zeros_like(conv)
                    reject = conv & ~cert_ok                   # look converged but fail cert -> maybe forward-stiff
                    if not last_tier and bool(reject.any()):
                        resid = forward_resid(m, sub, PI_CUR)
                        defer = reject & (resid > FWD_TOL)     # forward under-converged at this pi -> escalate
                    if bool(drop.any()) or bool(defer.any()):
                        if bool(drop.any()):
                            theta.index_copy_(0, active[drop], sub[drop]); was_dropped[active[drop]] = True
                            rebatch_log.append(dict(pi=PI_CUR, it=it, dropped=int(drop.sum()),
                                                    remain=int((~drop & ~defer).sum())))
                        if bool(defer.any()):
                            theta.index_copy_(0, active[defer], sub[defer]); carry = torch.cat([carry, active[defer]])
                            defer_log.append(dict(pi=PI_CUR, it=it, deferred=int(defer.sum()), to=PIS[pi_idx + 1],
                                                  stay=int((~drop & ~defer).sum()), resid_max=float(resid.max())))
                            print(f"  [pi{PI_CUR}] defer {int(defer.sum())} stiff (resid>{FWD_TOL:g}) -> "
                                  f"pi{PIS[pi_idx+1]}", flush=True)
                        keep = ~(drop | defer)
                        active = active[keep]; sub = sub[keep].clone()
                        if active.numel() == 0: break          # all done this tier; tier-end frees the (stale) m
                        del m; torch.cuda.empty_cache(); m = build_active(PI_CUR)
                        Hd = None; loss_ref = None; continue
                loss_ref = lv.clone()
            else:
                do_drop = frac > FRAC and bool(conv.any()) and not bool(conv.all())
                if do_drop and VERIFY_DROP:                    # verify converged subset at (CERT_PI, NEU_CERT) cold
                    _w = os.environ.pop("GPUREC_WARM_ADJOINT", None)
                    m.solver_options = sopts(CERT_PI, NEU_CERT)
                    conv = conv & (pgmax(sub, lg(m, sub)[1]) < TOL)
                    m.solver_options = sopts(PI_CUR, NEU_OPT)
                    if _w: os.environ["GPUREC_WARM_ADJOINT"] = _w
                    do_drop = bool(conv.any()) and not bool(conv.all())
                if do_drop:
                    theta.index_copy_(0, active[conv], sub[conv]); was_dropped[active[conv]] = True
                    active = active[~conv]; sub = sub[~conv].clone()
                    rebatch_log.append(dict(pi=PI_CUR, it=it, dropped=int(conv.sum()), remain=int(active.numel())))
                    had_drop = True; since_drop = 0
                    del m; torch.cuda.empty_cache(); m = build_active(PI_CUR)
                    Hd = None; loss_ref = None; continue
                since_drop += 1
                if not last_tier and ((had_drop and since_drop >= ESC_PATIENCE) or since_drop >= STUCK_MAX):
                    print(f"  [pi{PI_CUR}] stalled, {active.numel()} stiff -> escalate to pi{PIS[pi_idx+1]}", flush=True)
                    break
                loss_ref = lv.clone()
        if it % HESS_EVERY == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
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
    if active.numel() > 0: theta.index_copy_(0, active, sub)    # carry the residual theta into the next tier
    if REBATCH_RESID and active.numel() > 0: carry = torch.cat([carry, active])  # leftover -> next tier (safety net)
    del m; torch.cuda.empty_cache()
os.environ.pop("GPUREC_WARM_ADJOINT", None)
opt_s = time.perf_counter() - t0

# ---- final cert at (CERT_PI, NEU_CERT) COLD over ALL families -----------------------------------
tc = time.perf_counter()
mfull = build(fam_paths, CERT_PI, NEU_CERT)
_, g = lg(mfull, theta); pg = pgmax(theta, g)
H = torch.zeros(F_all, 3, 3, device=DEV, dtype=DT)
for j in range(3):
    tp = theta.clone(); tp[:, j] += FD_EPS; _, gp = lg(mfull, tp)
    tm = theta.clone(); tm[:, j] -= FD_EPS; _, gm = lg(mfull, tm)
    H[:, :, j] = (gp - gm) / (2 * FD_EPS)
H = 0.5 * (H + H.transpose(1, 2)); lam_min = torch.linalg.eigvalsh(H)[:, 0]
at_lo = (theta <= TH_LO + 1e-6); at_hi = (theta >= TH_HI - 1e-6); bound_active = (at_lo | at_hi).any(dim=1)
conv = pg < TOL; pd = lam_min > TOL; total = time.perf_counter() - t0
premature = int((was_dropped & ~conv).sum())
print(f"\n{'='*72}\nWARM + CONVERGENCE-REBATCH  ({DATASET} F={F_all}, pi-tiers={PIS} cert_pi={CERT_PI} neu_opt={NEU_OPT}-warm)\n{'='*72}", flush=True)
for r in rebatch_log:
    print(f"  drop @pi{r['pi']:3d} it{r['it']:3d}: dropped {r['dropped']:5d} -> {r['remain']:5d} remain", flush=True)
for r in defer_log:
    print(f"  defer @pi{r['pi']:3d} it{r['it']:3d}: {r['deferred']:5d} stiff -> pi{r['to']} ({r['stay']} stay)  "
          f"resid_max={r['resid_max']:.2e}", flush=True)
print(f"  drops={len(rebatch_log)}  total dropped mid-run={int(was_dropped.sum())}  optimize={opt_s:.0f}s  cert={time.perf_counter()-tc:.0f}s", flush=True)
print(f"  CONVERGED (|Pg|<{TOL}) = {int(conv.sum())}/{F_all}   |Pg|max={float(pg.max()):.2e}", flush=True)
print(f"  interior PD = {int((conv & pd & ~bound_active).sum())}   bound-active = {int(bound_active.sum())}   "
      f"unconverged = {int((~conv).sum())}   (of which dropped-prematurely = {premature})", flush=True)
print(f"  TOTAL = {total:.0f}s", flush=True)
R = dict(dataset=DATASET, F=F_all, pis=PIS, cert_pi=CERT_PI, neu_opt=NEU_OPT, neu_cert=NEU_CERT, conv_tol=CONV_TOL, frac=FRAC,
         opt_s=opt_s, total_s=total, n_conv=int(conv.sum()), n_interior_pd=int((conv & pd & ~bound_active).sum()),
         n_bound_active=int(bound_active.sum()), n_unconv=int((~conv).sum()), n_dropped=int(was_dropped.sum()),
         n_premature=premature, rebatches=rebatch_log, rebatch_resid=REBATCH_RESID, fwd_tol=FWD_TOL,
         improve_frac=IMPROVE_FRAC, defers=defer_log)
OUT = os.environ.get("OUT_JSON")
if OUT:
    torch.save(dict(theta=theta.cpu(), lam_min=lam_min.cpu(), pg=pg.cpu(), was_dropped=was_dropped.cpu()), OUT.replace(".json", "_theta.pt"))
    with open(OUT, "w") as fh: json.dump(R, fh, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"  saved -> {OUT}", flush=True)
