"""Unit test for cg_witness on a synthetic indefinite system with known spectrum (CPU, no GPU).

    python newton/test_witness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from newton.cg import cg_witness  # noqa: E402


def main():
    torch.manual_seed(0)
    n = 60
    # known spectrum: lam_min = -2, a flat cluster near 0, well-separated top
    eigs = torch.cat([
        torch.tensor([-2.0, -0.5]),
        torch.linspace(1e-4, 1e-2, 40),
        torch.linspace(1.0, 50.0, 17),
        torch.tensor([100.0]),
    ]).double()
    assert eigs.numel() == n
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    H = Q @ torch.diag(eigs) @ Q.T
    H = 0.5 * (H + H.T)
    lam_min = float(eigs.min())
    b = torch.randn(n, dtype=torch.float64)

    ok = True

    # (iii) lam > |lam_min|: witness must never fire; solve converges
    lam = abs(lam_min) * 1.2
    x, it, status, cert = cg_witness(lambda v: H @ v + lam * v, b, tol=1e-10, max_iter=500)
    res = float((H @ x + lam * x - b).norm())
    t3 = status == "converged" and cert is None and res < 1e-8
    print(f"[PD case]   lam={lam:.2f}: status={status} iters={it} residual={res:.2e} "
          f"{'PASS' if t3 else 'FAIL'}")
    ok &= t3

    # (i) lam < |lam_min|: witness fires; certificate is valid: lam < (lam - cert) <= |lam_min|
    lam = 0.5
    x, it, status, cert = cg_witness(lambda v: H @ v + lam * v, b, tol=1e-10, max_iter=500)
    need = lam - cert if cert is not None else float("nan")
    t1 = status == "neg_curv" and cert is not None and cert <= 0 and lam < need <= abs(lam_min) + 1e-9
    print(f"[indefinite] lam={lam:.2f}: status={status} cert={cert if cert is None else f'{cert:.4f}'} "
          f"certified |d^T H d|/|d|^2={need:.4f} (true |lam_min|={abs(lam_min):.2f}) "
          f"{'PASS' if t1 else 'FAIL'}")
    ok &= t1

    # (ii) bump loop lam <- nu*(lam - cert) terminates in a few bumps with a PD solve
    nu, lam, bumps = 1.5, 0.05, 0
    for bumps in range(1, 11):
        x, it, status, cert = cg_witness(lambda v: H @ v + lam * v, b, tol=1e-10, max_iter=500)
        if status != "neg_curv":
            break
        lam = nu * (lam - cert)
    res = float((H @ x + lam * x - b).norm())
    # the climb is geometric in the certified magnitude, so a 40x under-damped start needs
    # ~log_nu-ish(40) ~ 8 solves; real use starts at sigma*lam_max which is never that far off
    t2 = status == "converged" and bumps <= 8 and res < 1e-8 and lam > abs(lam_min)
    print(f"[bump loop]  start lam=0.05 -> {bumps} solve(s), final lam={lam:.3f} "
          f"status={status} residual={res:.2e} {'PASS' if t2 else 'FAIL'}")
    ok &= t2

    print(f"\nWITNESS TEST: {'ALL PASS' if ok else 'FAILURES'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
