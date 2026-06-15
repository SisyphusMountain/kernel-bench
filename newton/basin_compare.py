"""Compare two specieswise basins (part of the basin investigation, 2026-06-15; see
newton/_specieswise_basin_findings.md). Two analyses of A=137461 (rate-0.15) vs B=137384 (rate-0.25):

  (1) per-row DTL-probability diff (CPU): how many rows moved, which event probs, category flips;
  (2) NLL row-attribution (GPU): copy the most-changed rows A->B and watch the NLL -- is the gap
      localized or coordinated?

Result (2026-06-15): the change is DIFFUSE and COORDINATED -- ~61% of rows shift, only 2.9% flip
dominant category (mostly toward pD/duplication), and NO subset of rows carries the gap (copying the
top-100 rows makes the fit WORSE, -26%); the 77 NLL only appears once ~most of the rate field moves
together, because the reconciliation likelihood couples all states through the tree.

Run from repo root:  python -m newton.basin_compare
"""
import math
import os
os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")
import torch

LN2 = math.log(2.0); S = 1331; CATS = ["ref", "pS", "pD", "pL"]
CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
A_PATH = os.environ.get("CMP_A", os.path.join(CKPT_DIR, "basin_137461.pt"))
B_PATH = os.environ.get("CMP_B", os.path.join(CKPT_DIR, "specieswise_best_137384.pt"))


def probs(theta):
    logits = torch.cat([torch.zeros(theta.shape[0], 1, device=theta.device), theta], dim=1)
    return torch.softmax(logits * LN2, dim=1)


thA = torch.load(A_PATH, map_location="cpu", weights_only=False)["theta"].reshape(S, 3).float()
thB = torch.load(B_PATH, map_location="cpu", weights_only=False)["theta"].reshape(S, 3).float()
pA, pB = probs(thA), probs(thB)
dnorm = (thB - thA).norm(dim=1)

print("=== (1) per-row DTL-probability diff (CPU) ===")
print(f"  median ||dtheta||={dnorm.median():.3f}  rows moved >1: {int((dnorm>1).sum())} "
      f"({(dnorm>1).float().mean()*100:.1f}%)")
ss = (dnorm**2); cum = ss[ss.argsort(descending=True)].cumsum(0) / ss.sum()
print(f"  top-50 rows hold {cum[49]*100:.1f}% of squared movement; top-100 {cum[99]*100:.1f}%")
for j, nm in enumerate(CATS):
    print(f"  {nm}: mean|dp|={(pB[:,j]-pA[:,j]).abs().mean():.4e} max|dp|={(pB[:,j]-pA[:,j]).abs().max():.3f}")
flip = (pA.argmax(1) != pB.argmax(1))
print(f"  rows that flipped dominant category: {int(flip.sum())} ({flip.float().mean()*100:.1f}%); "
      f"of those -> pD: {int((flip & (pB.argmax(1)==2)).sum())}")

if not torch.cuda.is_available():
    print("\n(no CUDA -> skipping NLL attribution)"); raise SystemExit(0)

from newton.vg import load_problem, forward_solve, free_cuda_cache_if_tight
DEV = "cuda"
cap, static, theta0, cw = load_problem("666x80", DEV)
so = static.solver_options; so.pi_iters, so.neumann_terms = 128, 64
cwf = cw.float().contiguous()
thA, thB = thA.to(DEV), thB.to(DEV)


def nll(th):
    free_cuda_cache_if_tight()
    return float(forward_solve(static, th.reshape(S, 3), cwf)[0])


LA, LB = nll(thA), nll(thB)
order = (thB - thA).norm(dim=1).argsort(descending=True)
print(f"\n=== (2) NLL attribution (GPU)  L_A={LA:.2f} L_B={LB:.2f} gap={LA-LB:+.2f} ===")
print("  copy top-K most-changed rows A->B, rest stay A:")
for K in (10, 50, 100, 250, 500, 800, S):
    th = thA.clone(); th[order[:K]] = thB[order[:K]]
    L = nll(th)
    print(f"    K={K:4d}: NLL={L:.2f}  recovered {(LA-L)/(LA-LB)*100:+5.1f}% of the gap")
