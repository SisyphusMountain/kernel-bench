"""Part of the specieswise basin investigation (2026-06-15); see
newton/_specieswise_basin_findings.md for context. Run from the repo root with
`python -m newton.gauge_audit [checkpoint.pt]`.

First audit: does the theta[S,3] parameterization carry an exact softmax gauge (a per-row 1-vector
null direction, => up to S=1331 artificial flat directions), or is it already gauge-fixed?

extract_parameters does: logits = cat([0, theta], -1) (4 categories, FIRST PINNED TO 0);
p = softmax2(logits). The friend's gauge argument (add c*1 to a full softmax row leaves p invariant)
requires ALL category logits to be free. Here one is pinned -> shifting the 3 free logits up by c
does NOT preserve p (the reference category's prob moves). Test it directly.

Tests at the best checkpoint:
 (1) per-row all-ones direction u (u[s,:]=1): exact-gauge => f(theta + c*u) constant, g.u=0, ||Hu||=0.
 (2) Rayleigh quotient u^T H u / ||u||^2 and ||Hu||/||u|| vs a random direction (scale reference).
 (3) finite-difference f along u for several c (the cleanest exactness test).
"""
import os
import sys
os.environ["NEWTON_TANGENT_SELF_ITERS"] = "64"
import torch
from newton.vg import load_problem, forward_solve, make_value_and_grad, free_cuda_cache_if_tight
from newton.hvp_exact import make_exact_hvp

CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
DEV = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CKPT_DIR, "old_basin_137466.pt")
cap, static, theta0, cw = load_problem("666x80", DEV)
S = int(static.state_helpers["S"]); p = 3 * S
so = static.solver_options; so.pi_iters, so.neumann_terms = 128, 64
cwf = cw.float().contiguous()
theta = torch.load(CKPT, map_location=DEV)["theta"].to(DEV).reshape(S, 3).float().contiguous()
f = make_value_and_grad(static, cwf, grad_avg_K=2)
L0, g, _, _ = f(theta.reshape(-1))
print(f"ckpt={CKPT}  L0={float(L0):.5f}  ||g||={float(g.norm()):.4e}  S={S} p={p}", flush=True)

# per-row all-ones gauge direction (each row shifted by the same amount within its 3 free logits)
u = torch.ones(S, 3, device=DEV, dtype=torch.float32).reshape(-1)
u = u / u.norm()
# a random reference direction
gen = torch.Generator(device=DEV).manual_seed(0)
v = torch.randn(p, generator=gen, device=DEV, dtype=torch.float32); v = v / v.norm()

free_cuda_cache_if_tight()
_, sv = forward_solve(static, theta, cwf)
hvp = make_exact_hvp(static, theta, cwf, sv)
Hu = hvp(u.clone()); Hv = hvp(v.clone())
print("\n--- gauge direction u = per-row all-ones (normalized) ---", flush=True)
print(f"  g . u            = {float(torch.dot(g.float(), u)):+.6e}   (exact gauge => 0)", flush=True)
print(f"  ||H u|| / ||u||  = {float(Hu.norm()):.6e}   (exact gauge => 0)", flush=True)
print(f"  u^T H u / ||u||^2= {float(torch.dot(u, Hu)):+.6e}   (exact gauge => 0)", flush=True)
print("\n--- random reference direction v (scale for comparison) ---", flush=True)
print(f"  ||H v|| / ||v||  = {float(Hv.norm()):.6e}", flush=True)
print(f"  v^T H v / ||v||^2= {float(torch.dot(v, Hv)):+.6e}", flush=True)

print("\n--- finite-difference f along the gauge direction (exact gauge => Df == 0) ---", flush=True)
for c in (-0.1, -0.03, -0.01, 0.01, 0.03, 0.1):
    free_cuda_cache_if_tight()
    Lc = float(forward_solve(static, (theta.reshape(-1) + c * u).reshape(S, 3), cwf)[0])
    print(f"  c={c:+.3f}:  f-f0 = {Lc-float(L0):+.6e}", flush=True)

# also test a SINGLE row's all-ones (one species) to be thorough about the per-row claim
u1 = torch.zeros(S, 3, device=DEV, dtype=torch.float32); u1[0, :] = 1.0; u1 = u1.reshape(-1) / 3 ** 0.5
print("\n--- single-row (species 0) all-ones FD (exact per-row gauge => 0) ---", flush=True)
for c in (-0.2, 0.2, 1.0):
    free_cuda_cache_if_tight()
    Lc = float(forward_solve(static, (theta.reshape(-1) + c * u1).reshape(S, 3), cwf)[0])
    print(f"  c={c:+.3f}:  f-f0 = {Lc-float(L0):+.6e}", flush=True)
print("\nDONE", flush=True)
