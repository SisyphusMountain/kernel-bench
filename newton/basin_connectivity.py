"""Mode-connectivity search between the two specieswise basins (137461 rate-0.15 vs 137384 rate-0.25).
Part of the basin investigation (2026-06-15); see newton/_specieswise_basin_findings.md.

The straight line between them has a ~+85 NLL barrier (newton/basin_interp.py). This asks the sharper
question: is there a CURVED low path? Fit a quadratic Bezier theta(t) = (1-t)^2 th_A + 2t(1-t) mid +
t^2 th_B with FIXED endpoints and optimize the single control point `mid` to minimize the mean NLL
along the path (Garipov et al. 2018). If the optimized barrier collapses toward the endpoint level,
the basins are the SAME mode (connected); if it stays high, they are genuinely separate modes.

Run from repo root:  python -m newton.basin_connectivity
"""
import os
os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")
import numpy as np
import torch
from newton.vg import load_problem, forward_solve, make_value_and_grad, free_cuda_cache_if_tight

DEV = "cuda"
CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
FIGDIR = os.path.join(os.path.dirname(__file__), "_figures")
os.makedirs(FIGDIR, exist_ok=True)
STEPS = int(os.environ.get("CONN_STEPS", "60"))
LR = float(os.environ.get("CONN_LR", "0.3"))

cap, static, theta0, cw = load_problem("666x80", DEV)
S = int(static.state_helpers["S"]); so = static.solver_options; so.pi_iters, so.neumann_terms = 128, 64
cwf = cw.float().contiguous()
f = make_value_and_grad(static, cwf, grad_avg_K=1)
# A = 137461 (rate-0.15), B = 137384 (rate-0.25). 137461 lives in /tmp; fall back to committed if absent.
A_path = "/tmp/claude-1000/followup_best.pt"
A_path = A_path if os.path.exists(A_path) else os.path.join(CKPT_DIR, "old_basin_137466.pt")
thA = torch.load(A_path, map_location=DEV, weights_only=False)["theta"].reshape(S, 3).float()
thB = torch.load(os.path.join(CKPT_DIR, "specieswise_best_137384.pt"), map_location=DEV, weights_only=False)["theta"].reshape(S, 3).float()


def nll(th):
    free_cuda_cache_if_tight()
    return float(forward_solve(static, th.reshape(S, 3), cwf)[0])


def bezier(t, mid):
    return (1 - t) ** 2 * thA + 2 * t * (1 - t) * mid + t ** 2 * thB


LA, LB = nll(thA), nll(thB)
TS = np.linspace(0.1, 0.9, 9)                      # interior grid for the path objective
mid = (0.5 * thA + 0.5 * thB).clone().requires_grad_(True)
opt = torch.optim.Adam([mid], lr=LR)
print(f"L_A(461)={LA:.2f}  L_B(384)={LB:.2f}  STEPS={STEPS} lr={LR}", flush=True)
print(f"linear barrier (mid at start) = {nll(0.5*thA+0.5*thB):.2f}  (+{nll(0.5*thA+0.5*thB)-LA:.1f} vs A)\n", flush=True)

for step in range(STEPS):
    opt.zero_grad()
    g_mid = torch.zeros_like(mid)
    obj, maxL = 0.0, 0.0
    md = mid.detach()
    for t in TS:
        th_t = bezier(float(t), md)
        L, g, _, _ = f(th_t.reshape(-1))
        obj += L; maxL = max(maxL, L)
        g_mid += (2 * t * (1 - t)) * g.reshape(S, 3)
    mid.grad = g_mid / len(TS)
    opt.step()
    if step % 5 == 0 or step == STEPS - 1:
        print(f"  step {step:3d}: mean NLL on path={obj/len(TS):.2f}  max(interior)={maxL:.2f}", flush=True)

# final fine-grid evaluation of the optimized Bezier
md = mid.detach()
grid = np.round(np.linspace(-0.1, 1.1, 49), 4)
Ls = np.array([nll(bezier(float(t), md)) for t in grid])
seg = (grid >= 0) & (grid <= 1)
ib = np.where(seg)[0][np.argmax(Ls[seg])]
print(f"\n=== optimized curved path ===", flush=True)
print(f"  barrier (max on [0,1]) at t={grid[ib]:.3f}: NLL={Ls[ib]:.2f}", flush=True)
print(f"  height above A(461) = {Ls[ib]-LA:+.2f}  (linear path was +84.7)", flush=True)
verdict = ("CONNECTED (curved path nearly removes the barrier -> same mode)"
           if Ls[ib] - LA < 15 else
           "SEPARATE modes (barrier persists even on the optimized curve)")
print(f"  VERDICT: {verdict}", flush=True)

torch.save({"theta": md.cpu(), "barrier_nll": float(Ls[ib])}, os.path.join(CKPT_DIR, "connectivity_mid.pt"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(grid, Ls, "-o", ms=3, label="optimized Bezier path")
    plt.axhline(LA, color="g", ls=":", lw=1); plt.axhline(LB, color="r", ls=":", lw=1)
    plt.axvline(0, color="g", ls="--", lw=1, label="137461 (A)"); plt.axvline(1, color="r", ls="--", lw=1, label="137384 (B)")
    plt.xlabel("path parameter t"); plt.ylabel("NLL")
    plt.title("Mode-connectivity: optimized curved path between the basins")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "basin_connectivity.png"), dpi=110)
    print(f"  saved plot -> newton/_figures/basin_connectivity.png", flush=True)
except Exception as e:
    print(f"  (matplotlib unavailable: {e})", flush=True)
print("DONE", flush=True)
