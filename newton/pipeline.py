"""Three-stage fit pipeline: Adam (bulk, fp32) -> ridge TR-Newton polish (fp64) -> Fisher/Laplace
uncertainty.

    python newton/pipeline.py --size small
    python newton/pipeline.py --size 1007x64 --adam-steps 500 --skip-fisher

Stage 1  Adam on the fp32 forward/backward (warm-started E), stopping when the gradient is
         relatively small or the loss stops changing.
Stage 2  Lanczos at the Adam endpoint -> lam = -min(lam_min,0) + sigma*lam_max (the measured
         spectrum here is flat/indefinite at the optimum: 39/357 negative, ~250/357 near-zero on
         `small`, so unridged Newton cannot converge quadratically); then trust-region Newton on
         the MAP objective F = L + lam/2 ||x - x_adam||^2 with fp64 FD-of-gradient HVPs.
Stage 3  Observed information H (dense, p FD-HVP columns) at the MAP point; Laplace covariance
         (H + lam I)^-1; per-parameter std; flags parameters whose uncertainty is prior-dominated
         (the data does not constrain them). Saves everything to data/<size>/newton_fit.pt.

All stages are matrix-free except stage 3, which is O(p) HVPs = O(2p) gradient evals — fine
offline up to p ~ a few thousand.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# kbench's backward gates its scratch on (driver-free - reserve); the default 1 GiB reserve is
# conservative and rejects feasible fp64 runs on the big fixtures. Respect a user override.
os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB", "0.25")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from newton.vg import DATA, free_cuda_cache_if_tight, load_problem, make_value_and_grad  # noqa: E402
from newton.newton_cg import _fd_hessian_hvp, newton_lanczos, newton_tr  # noqa: E402
from newton.cg import lanczos_extremes  # noqa: E402


def adam(static, theta0, col_weights, *, lr=0.02, betas=(0.9, 0.999), eps=1e-8,
         max_steps=2000, window=20, loss_rtol=1e-5, gtol_rel=0.05, log_every=50, verbose=True):
    """Adam until ||g|| < gtol_rel*||g0|| or the loss is flat over `window` steps."""
    S = int(static.state_helpers["S"])
    f = make_value_and_grad(static, col_weights)
    x = theta0.reshape(-1).clone()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    b1, b2 = betas
    warm, g0, hist = None, None, []
    for t in range(1, int(max_steps) + 1):
        free_cuda_cache_if_tight()
        loss, g, sv, warm = f(x, warm_E=warm)
        # keep only the small E for warm-starting; holding the full intermediates (pi/pibar are
        # ~GBs on the big fixtures) across iterations doubles peak memory
        del sv
        gn = float(g.norm())
        g0 = gn if g0 is None else g0
        hist.append({"step": t, "loss": loss, "gnorm": gn})
        if verbose and (t == 1 or t % log_every == 0):
            print(f"[adam {t:5d}] loss={loss:.4f}  ||g||={gn:.4e}")
        if t >= window:
            recent = [h["loss"] for h in hist[-window:]]
            flat = (max(recent) - min(recent)) <= loss_rtol * max(1.0, abs(loss))
            if gn <= gtol_rel * g0 or flat:
                why = "gradient small" if gn <= gtol_rel * g0 else "loss flat"
                if verbose:
                    print(f"[adam {t:5d}] stop ({why}): loss={loss:.4f}  ||g||={gn:.4e}")
                break
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        x = x - lr * mh / (vh.sqrt() + eps)
    return x.reshape(S, 3), hist, warm


def auto_lambda(static, theta, col_weights, *, m=40, sigma=0.01, verbose=True):
    """lam = -min(lam_min,0) + sigma*lam_max from a matrix-free Lanczos at theta (fp64)."""
    vg = make_value_and_grad(static, col_weights)
    x = theta.reshape(-1).double().contiguous()
    out = vg(x)
    warm = out[3]
    del out  # do not pin the saved intermediates across the Lanczos loop
    hvp = _fd_hessian_hvp(vg, x, warm)
    p = x.numel()
    t0 = time.time()
    lo, hi = lanczos_extremes(lambda q: hvp(q).double(), p, m=m, device=str(x.device))
    lam = -min(lo, 0.0) + sigma * hi
    if verbose:
        print(f"[lanczos] m={m}  lam_min~{lo:+.4e} (upper bound)  lam_max~{hi:.2f}  "
              f"-> lam={lam:.4f}  ({time.time()-t0:.0f}s)")
    return lam, lo, hi


def fisher_laplace(static, theta_hat, col_weights, lam, *, verbose=True):
    """Dense observed information H at theta_hat (FD-HVP columns, fp64), Laplace covariance
    (H + lam I)^-1, per-parameter std. Returns dict of tensors (CPU)."""
    vg = make_value_and_grad(static, col_weights)
    x = theta_hat.reshape(-1).double().contiguous()
    p = x.numel()
    out = vg(x)
    loss, g, warm = out[0], out[1], out[3]
    del out  # do not pin the saved intermediates across the column loop
    hvp = _fd_hessian_hvp(vg, x, warm)
    t0 = time.time()
    H = torch.zeros((p, p), dtype=torch.float64, device=x.device)
    e = torch.zeros(p, dtype=torch.float64, device=x.device)
    for i in range(p):
        e.zero_(); e[i] = 1.0
        H[:, i] = hvp(e).double()
        if verbose and (i + 1) % max(1, p // 5) == 0:
            print(f"[fisher] column {i+1}/{p}  ({time.time()-t0:.0f}s)")
    H = 0.5 * (H + H.T)
    ev = torch.linalg.eigvalsh(H)
    A = H + lam * torch.eye(p, dtype=torch.float64, device=x.device)
    cov = torch.cholesky_inverse(torch.linalg.cholesky(A))
    std = cov.diagonal().sqrt()
    prior_var = 1.0 / lam
    prior_dominated = cov.diagonal() >= 0.9 * prior_var  # data adds (almost) nothing here
    if verbose:
        lam1 = float(ev[-1])
        print(f"[fisher] H built in {time.time()-t0:.0f}s  lam_max={lam1:.2f}  "
              f"lam_min={float(ev[0]):+.4e}  n_neg={int((ev < 0).sum())}/{p}  "
              f"n_flat(<0.1%lam_max)={int((ev < 1e-3 * lam1).sum())}/{p}")
        print(f"[fisher] std: min={float(std.min()):.4f}  median={float(std.median()):.4f}  "
              f"max={float(std.max()):.4f}  (prior floor 1/sqrt(lam)={prior_var**0.5:.4f})")
        print(f"[fisher] prior-dominated (unidentified) parameters: {int(prior_dominated.sum())}/{p}")
    S = int(static.state_helpers["S"])
    return {
        "loss": float(loss), "gnorm": float(g.norm()), "lam": float(lam),
        "H": H.cpu(), "eigvals": ev.cpu(), "cov": cov.cpu(),
        "std": std.reshape(S, 3).cpu(), "prior_dominated": prior_dominated.reshape(S, 3).cpu(),
    }


def run(args):
    cap, static, theta, col_weights = load_problem(args.size)
    S = int(static.state_helpers["S"]); p = 3 * S
    print(f"=== pipeline [{args.size}]  S={S}  p={p} ===")

    # stage 1: Adam (fp32)
    t0 = time.time()
    theta_a, hist_a, _ = adam(
        static, theta, col_weights, lr=args.adam_lr, max_steps=args.adam_steps,
        window=args.adam_window, loss_rtol=args.adam_rtol, gtol_rel=args.adam_gtol_rel,
    )
    print(f"[stage 1] adam: {len(hist_a)} steps, {time.time()-t0:.0f}s  "
          f"loss {hist_a[0]['loss']:.2f} -> {hist_a[-1]['loss']:.2f}")

    # stage 2: lambda + ridge TR-Newton polish (fp64)
    t0 = time.time()
    cw64 = col_weights.double()
    lanczos_m = args.lanczos_m if args.polisher == "tr" else min(args.lanczos_m, 10)
    if args.lam is not None:
        lam = float(args.lam)
        print(f"[stage 2] lam={lam} (user)")
    else:
        # tr: full rule (needs a trustworthy lam_min -> m~40). lanczos polisher: the witness
        # self-corrects negativity at the solver level, so m~10 (lam_max only) suffices.
        lam, _, _ = auto_lambda(static, theta_a, cw64, m=lanczos_m, sigma=args.sigma)
    if args.polisher == "lanczos":
        theta_hat, hist_p = newton_lanczos(
            static, theta_a.double(), cw64, lam=lam, sigma=args.sigma, lanczos_m=lanczos_m,
            max_newton=args.polish_steps, gtol=args.gtol, max_cg=args.max_cg,
        )
        nfires = sum(len(h.get("witness_certs", [])) for h in hist_p)
        print(f"[stage 2] witness fires: {nfires}")
    else:
        theta_hat, hist_p = newton_tr(
            static, theta_a.double(), cw64, curvature="fd_hessian", lam=lam,
            max_newton=args.polish_steps, gtol=args.gtol, delta0=1.0, max_cg=args.max_cg,
        )
    print(f"[stage 2] polish: {len(hist_p)} steps, {time.time()-t0:.0f}s  "
          f"||gF|| {hist_p[0]['gnorm']:.2e} -> {hist_p[-1]['gnorm']:.2e}")

    out = {"theta_hat": theta_hat.cpu(), "lam": lam,
           "adam_hist": hist_a, "polish_hist": hist_p, "size": args.size}

    # stage 3: Fisher / Laplace
    if args.skip_fisher:
        print("[stage 3] skipped (--skip-fisher)")
    else:
        t0 = time.time()
        est_min = 2 * p * 1.2 / 60
        print(f"[stage 3] building dense H: {p} HVP columns (~{est_min:.0f} min on small-class sizes)")
        out.update(fisher_laplace(static, theta_hat, cw64, lam))
        print(f"[stage 3] done in {time.time()-t0:.0f}s")
        std = out["std"]
        flat_idx = torch.nonzero(out["prior_dominated"])
        order = std.reshape(-1).argsort(descending=True)
        print("  top-5 most uncertain theta entries (state, col): "
              + ", ".join(f"({int(i // 3)},{int(i % 3)}) std={float(std.reshape(-1)[i]):.3f}"
                          for i in order[:5]))
        print(f"  unidentified (prior-dominated) entries: {flat_idx.shape[0]}/{p}")

    path = DATA / args.size / "newton_fit.pt"
    torch.save(out, path)
    print(f"\nsaved -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="small")
    ap.add_argument("--adam-lr", type=float, default=0.02)
    ap.add_argument("--adam-steps", type=int, default=2000)
    ap.add_argument("--adam-window", type=int, default=20)
    ap.add_argument("--adam-rtol", type=float, default=1e-5)
    ap.add_argument("--adam-gtol-rel", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=None, help="ridge strength (default: auto from Lanczos)")
    ap.add_argument("--sigma", type=float, default=0.01, help="lam = -min(lam_min,0)+sigma*lam_max")
    ap.add_argument("--lanczos-m", type=int, default=40)
    ap.add_argument("--polisher", default="tr", choices=["tr", "lanczos"])
    ap.add_argument("--polish-steps", type=int, default=12)
    ap.add_argument("--gtol", type=float, default=1e-3)
    ap.add_argument("--max-cg", type=int, default=40)
    ap.add_argument("--skip-fisher", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
