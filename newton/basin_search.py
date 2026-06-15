"""Part of the specieswise basin investigation (2026-06-15); see
newton/_specieswise_basin_findings.md for context. Run from the repo root with
`python -m newton.basin_search`.

Convergence audit confirmed 137385.88 (rate 0.25) is REAL, not a truncation artifact. The "stuck"
was initialization scale, not solver fidelity. Resume the sweep UPWARD to find where the basin bottoms
out, sample finely near the 0.20->0.25 threshold, and add repeats to gauge run-to-run variance
(~3 NLL seen). Each: Adam(400) -> L-BFGS(maxcor=100). Local fp32, full depth (pi=128/neumann=64)."""
import math
import os
import time
os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")
import torch
from newton.vg import load_problem, forward_solve, make_value_and_grad, free_cuda_cache_if_tight
from newton.optimize import first_order
from newton.baselines import lbfgs_scipy

CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
DEV = "cuda"
cap, static, theta0, cw = load_problem("666x80", DEV)
S = int(static.state_helpers["S"]); p = 3 * S
so = static.solver_options; so.pi_iters, so.neumann_terms = 128, 64
cwf = cw.float().contiguous()
f = make_value_and_grad(static, cwf, grad_avg_K=2)
t0 = time.perf_counter()


def nll_g(th):
    free_cuda_cache_if_tight()
    L, g, _, _ = f(th.reshape(-1))
    return float(L), float(g.norm())


def dive(theta_init, steps=400):
    free_cuda_cache_if_tight()
    _, _, _, th_best = first_order(static, theta_init, cwf, optimizer="adam", lr0=1.0, schedule="constant",
                                   max_steps=steps, verbose=False, t0_wall=t0, return_best=True, early_stop=False)
    free_cuda_cache_if_tight()
    out, _ = lbfgs_scipy(static, th_best.reshape(S, 3).float().contiguous(), cwf,
                         maxiter=2000, maxcor=100, dtype=torch.float32, verbose=False)
    return th_best, out.reshape(S, 3).float().contiguous()


def uniform(rate):
    return torch.full((S, 3), math.log2(rate), device=DEV, dtype=torch.float32).contiguous()


def jitter(rate, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return (uniform(rate) + 0.5 * torch.randn(S, 3, generator=gen, device=DEV, dtype=torch.float32)).contiguous()


INITS = [("rate_0.22", uniform(0.22)), ("rate_0.25", uniform(0.25)), ("rate_0.25b", uniform(0.25)),
         ("rate_0.28", uniform(0.28)), ("rate_0.30", uniform(0.30)), ("rate_0.35", uniform(0.35)),
         ("rate_0.40", uniform(0.40)), ("rate_0.50", uniform(0.50)), ("rate_0.70", uniform(0.70)),
         ("rate_1.00", uniform(1.00)),
         ("jit_0.25_a", jitter(0.25, 21)), ("jit_0.30_a", jitter(0.30, 22)), ("jit_0.30_b", jitter(0.30, 23))]

print(f"=== extended rate sweep (rate0.25 confirmed 137385.88 REAL; prior best 137466.06) ===", flush=True)
best = (1e18, None, None)
rows = []
for name, th0 in INITS:
    th_a, th_lb = dive(th0)
    La, _ = nll_g(th_a)
    L, gn = nll_g(th_lb)
    rng = (float(th_lb.min()), float(th_lb.max()))
    rows.append((name, La, L, gn, rng))
    flag = "  <== NEW BEST" if L < best[0] else ""
    print(f"  {name:12s} adam.best={La:11.4f} -> lbfgs={L:11.4f}  ||g||={gn:.3e}  th[{rng[0]:6.2f},{rng[1]:5.2f}]"
          f"  ({time.perf_counter()-t0:.0f}s){flag}", flush=True)
    if L < best[0]:
        best = (L, name, th_lb.clone())
        torch.save({"theta": th_lb.cpu(), "L": L, "src": name}, os.path.join(CKPT_DIR, "basin_search_best.pt"))

rows.sort(key=lambda r: r[2])
print("\n=== sorted by lbfgs raw NLL ===", flush=True)
for name, La, L, gn, rng in rows:
    print(f"  {name:12s} nll={L:11.4f}  ||g||={gn:.3e}", flush=True)
print(f"\nBEST = {best[0]:.4f} ({best[1]})  vs prior 137466.06 / rate0.25 137385.88", flush=True)
print(f"DONE ({time.perf_counter()-t0:.0f}s)", flush=True)
