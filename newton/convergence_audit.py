"""Part of the specieswise basin investigation (2026-06-15); see
newton/_specieswise_basin_findings.md for context. Run from the repo root with
`python -m newton.convergence_audit`.

Is the new ~137461 minimum REAL or a truncation artifact? Three audits at the best checkpoint:

(A) FORWARD/Pi convergence of the LOSS: loss vs pi_iters in {32..1024} and e_tol in {1e-6..1e-10}.
    The loss depends ONLY on the forward solve (pi fixed-point + E-step), not on neumann/tangent.
    If the loss is still moving past pi=128 by more than the ~5 NLL basin gaps we chase, the value
    is under-converged and the basin rankings are noise.

(B) GRADIENT/Neumann convergence: ||g|| and the gradient DIRECTION vs neumann_terms in {16..256}
    (pi fixed high). L-BFGS minimizes using the analytic (neumann) gradient; if that gradient is
    biased, L-BFGS stalls at the minimum of a TRUNCATED objective, not the true one. Self-consistency
    across neumann is the test: if g stops changing by neumann=64, it's converged.

(C) HIGH-FIDELITY re-optimization: L-BFGS from theta* at pi=256/neumann=128. If it descends BELOW
    137461, we were stuck due to truncation. If it holds, 137461 is real and the stall is geometric.

Local fp32 (loss quantization ~0.01 << the effects we test). Picks the lowest-loss checkpoint.
"""
import os
import time
os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")
import torch
from newton.vg import load_problem, forward_solve, make_value_and_grad, free_cuda_cache_if_tight
from newton.baselines import lbfgs_scipy

CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
DEV = "cuda"
cap, static, theta0, cw = load_problem("666x80", DEV)
S = int(static.state_helpers["S"]); p = 3 * S
so = static.solver_options
cwf = cw.float().contiguous()
t0 = time.perf_counter()


def set_solver(pi, neu, etol=None):
    so.pi_iters, so.neumann_terms = pi, neu
    if etol is not None and hasattr(so, "e_tol"):
        so.e_tol = etol


def loss_at(th, pi, etol=None):
    set_solver(pi, so.neumann_terms, etol)
    free_cuda_cache_if_tight()
    return float(forward_solve(static, th.reshape(S, 3), cwf)[0])


def grad_at(th, pi, neu):
    set_solver(pi, neu)
    f = make_value_and_grad(static, cwf, grad_avg_K=4)   # avg to suppress backward atomic noise
    free_cuda_cache_if_tight()
    _, g, _, _ = f(th.reshape(-1))
    return g.float()


# pick best checkpoint
cands = []
for path in (os.path.join(CKPT_DIR, "specieswise_best_137384.pt"),
             os.path.join(CKPT_DIR, "old_basin_137466.pt")):
    if os.path.exists(path):
        d = torch.load(path, map_location=DEV)
        th = d["theta"].to(DEV).reshape(S, 3).float().contiguous()
        cands.append((loss_at(th, 128), path, th))
cands.sort(key=lambda x: x[0])
L0, ckpt, theta = cands[0]
print(f"=== convergence audit @ {ckpt}  (pi128 loss={L0:.4f})  S={S} p={p} ===\n", flush=True)
print(f"  e_tol attr present: {hasattr(so, 'e_tol')}  default e_tol={getattr(so,'e_tol','n/a')}", flush=True)

# (A) forward/Pi convergence of the loss
print("\n=== (A) loss vs pi_iters (e_tol=default) -- ref = pi1024 ===", flush=True)
ref = loss_at(theta, 1024)
for pi in (32, 64, 128, 256, 512, 1024):
    L = loss_at(theta, pi)
    print(f"  pi={pi:5d}: loss={L:.4f}  (loss - pi1024) = {L-ref:+.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
if hasattr(so, "e_tol"):
    print("  -- e_tol sweep at pi=512 --", flush=True)
    for et in (1e-6, 1e-8, 1e-10):
        L = loss_at(theta, 512, et)
        print(f"  e_tol={et:.0e}: loss={L:.4f}", flush=True)
    so.e_tol = 1e-8

# (B) gradient/Neumann convergence
print("\n=== (B) gradient vs neumann_terms (pi=512) -- ||g|| and dir-change vs neumann=256 ===", flush=True)
gref = grad_at(theta, 512, 256)
for neu in (16, 32, 64, 128, 256):
    g = grad_at(theta, 512, neu)
    cos = float(torch.dot(g, gref) / (g.norm() * gref.norm() + 1e-30))
    rel = float((g - gref).norm() / (gref.norm() + 1e-30))
    print(f"  neumann={neu:4d}: ||g||={float(g.norm()):.4e}  cos(g,g256)={cos:.6f}  "
          f"||g-g256||/||g256||={rel:.4e}  ({time.perf_counter()-t0:.0f}s)", flush=True)

# (C) high-fidelity re-optimization
print("\n=== (C) high-fidelity L-BFGS from theta* (pi=256, neumann=128, maxcor=100) ===", flush=True)
set_solver(256, 128)
free_cuda_cache_if_tight()
th_hi, _ = lbfgs_scipy(static, theta, cwf, maxiter=1500, maxcor=100, dtype=torch.float32, verbose=False)
th_hi = th_hi.reshape(S, 3).float().contiguous()
L_hi_pi256 = loss_at(th_hi, 256)
L_hi_pi1024 = loss_at(th_hi, 1024)
L0_pi1024 = loss_at(theta, 1024)
print(f"  start  : pi1024 loss = {L0_pi1024:.4f}", flush=True)
print(f"  hi-fid : pi256  loss = {L_hi_pi256:.4f}   pi1024 loss = {L_hi_pi1024:.4f}", flush=True)
print(f"  delta vs start (pi1024) = {L_hi_pi1024 - L0_pi1024:+.4f}  -> "
      f"{'WAS STUCK on truncation (hi-fid goes lower)' if L_hi_pi1024 < L0_pi1024 - 0.2 else 'no further descent: 137461 is real'}",
      flush=True)
torch.save({"theta": th_hi.cpu(), "L_pi1024": L_hi_pi1024}, os.path.join(CKPT_DIR, "convergence_hifit.pt"))
print(f"\nDONE ({time.perf_counter()-t0:.0f}s)", flush=True)
