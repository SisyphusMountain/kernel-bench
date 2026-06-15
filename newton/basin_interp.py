"""Linear interpolation between two specieswise basins (part of the basin investigation, 2026-06-15;
see newton/_specieswise_basin_findings.md). Evaluate NLL along theta(a) = (1-a)*th_A + a*th_B for
a in [-0.25, 1.25] to see whether a barrier separates the basins. Forward solves only.

Default A = 137461 (rate-0.15 basin), B = 137384 (rate-0.25 basin). Run from repo root:
  python -m newton.basin_interp                      # straight line in THETA (log-odds) space
  INTERP_SPACE=prob python -m newton.basin_interp    # straight line in PROBABILITY (simplex) space

THETA-space result (2026-06-15): a clear barrier at a=0.65, NLL=137546 -> +84.7 above A, +161.8 above
B; both endpoints are wells (NLL rises for a<0 and a>1). The basins are NOT linearly connected.

PROB-space variant: interpolate the per-row 4-category probabilities p(a)=(1-a)p_A+a p_B linearly in
the simplex, then map back to theta via theta_j = log2(p_j / p_ref). Tests whether the theta-space
barrier is partly an artifact of the log-odds parameterization (which compresses the boundary, where
~half the rates live). Only defined on a in [0,1] (the convex blend stays in the simplex).
"""
import os
os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")
import numpy as np
import torch
from newton.vg import load_problem, forward_solve, free_cuda_cache_if_tight

DEV = "cuda"
CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
FIGDIR = os.path.join(os.path.dirname(__file__), "_figures"); os.makedirs(FIGDIR, exist_ok=True)
A_PATH = os.environ.get("INTERP_A", os.path.join(CKPT_DIR, "basin_137461.pt"))
B_PATH = os.environ.get("INTERP_B", os.path.join(CKPT_DIR, "specieswise_best_137384.pt"))

cap, static, theta0, cw = load_problem("666x80", DEV)
S = int(static.state_helpers["S"]); so = static.solver_options; so.pi_iters, so.neumann_terms = 128, 64
cwf = cw.float().contiguous()
thA = torch.load(A_PATH, map_location=DEV, weights_only=False)["theta"].reshape(S, 3).float()
thB = torch.load(B_PATH, map_location=DEV, weights_only=False)["theta"].reshape(S, 3).float()


def nll(th):
    free_cuda_cache_if_tight()
    return float(forward_solve(static, th.reshape(S, 3), cwf)[0])


SPACE = os.environ.get("INTERP_SPACE", "theta")   # "theta" (log-odds) or "prob" (simplex)
TAG = "prob" if SPACE == "prob" else "theta"


def _probs(th):
    logits = torch.cat([torch.zeros(th.shape[0], 1, device=th.device), th], dim=1)
    return torch.softmax(logits * float(torch.log(torch.tensor(2.0))), dim=1)


if SPACE == "prob":
    pA, pB = _probs(thA), _probs(thB)
    alphas = np.round(np.linspace(0.0, 1.0, 41), 4)          # prob-line valid only in the simplex

    def interp(a):
        p = ((1 - a) * pA + a * pB).clamp_min(1e-30)
        return torch.log2(p[:, 1:]) - torch.log2(p[:, 0:1])  # theta_j = log2(p_j / p_ref)
else:
    alphas = np.round(np.linspace(-0.25, 1.25, 61), 4)

    def interp(a):
        return (1 - a) * thA + a * thB


Ls = np.array([nll(interp(a)) for a in alphas])
LA, LB = Ls[np.argmin(np.abs(alphas))], Ls[np.argmin(np.abs(alphas - 1))]
seg = (alphas >= 0) & (alphas <= 1); ib = np.where(seg)[0][np.argmax(Ls[seg])]
print(f"space={SPACE}  L_A={LA:.2f}  L_B={LB:.2f}", flush=True)
print(f"barrier at a={alphas[ib]:.3f}: NLL={Ls[ib]:.2f}  (+{Ls[ib]-LA:.1f} vs A, +{Ls[ib]-LB:.1f} vs B)", flush=True)
print(f"  -> {'NO barrier' if Ls[ib] <= max(LA, LB) + 0.5 else 'BARRIER present'}", flush=True)
for a, L in zip(alphas, Ls):
    print(f"  a={a:+.3f}  NLL={L:.2f}", flush=True)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5)); plt.plot(alphas, Ls, "-o", ms=3)
    plt.axvline(0, color="g", ls="--", lw=1, label="A"); plt.axvline(1, color="r", ls="--", lw=1, label="B")
    plt.xlabel("alpha (0=A, 1=B)"); plt.ylabel("NLL")
    plt.title(f"Linear interpolation between basins ({SPACE} space)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(FIGDIR, f"interp_{TAG}_461_384.png")
    plt.savefig(out, dpi=110)
    print(f"saved -> {out}", flush=True)
except Exception as e:
    print(f"(matplotlib unavailable: {e})", flush=True)
