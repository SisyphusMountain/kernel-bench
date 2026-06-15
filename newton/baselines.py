"""First-order / quasi-Newton baselines to calibrate the Newton-CG optimizer."""

from __future__ import annotations

import numpy as np
import torch

from newton.vg import make_value_and_grad


def gd(static, theta0, col_weights, *, lr=3e-3, steps=300, verbose=False):
    """Plain gradient descent (fp32). Returns (theta, history)."""
    S = int(static.state_helpers["S"])
    f = make_value_and_grad(static, col_weights)
    x = theta0.reshape(-1).clone()
    warm = None
    hist = []
    for k in range(steps):
        loss, g, sv, warm = f(x, warm_E=warm)
        gnorm = float(g.norm())
        hist.append({"step": k, "loss": loss, "gnorm": gnorm})
        if verbose and k % max(1, steps // 20) == 0:
            print(f"[gd {k:4d}] loss={loss:.4f} ||g||={gnorm:.4e}")
        x = x - lr * g
    return x.reshape(S, 3), hist


def lbfgs_scipy(static, theta0, col_weights, *, maxiter=100, verbose=False, dtype=None):
    """L-BFGS-B (scipy outer loop in fp64) on value_and_grad. ``dtype`` is the solve/gradient
    precision (default = ``theta0.dtype``). Pass ``torch.float32`` for the big fixtures: the fp64
    forward/HVP OOMs on 24 GB and the geometry is dtype-independent. Returns (theta, history)."""
    from scipy.optimize import minimize

    dt = dtype or theta0.dtype
    S = int(static.state_helpers["S"])
    cw = col_weights.to(dt)
    f = make_value_and_grad(static, cw)
    dev = theta0.device
    state = {"warm": None, "n": 0, "hist": []}

    def fun(x_np):
        x = torch.tensor(x_np, device=dev, dtype=dt)
        loss, g, sv, warm = f(x, warm_E=state["warm"])
        state["warm"] = warm
        state["n"] += 1
        state["hist"].append({"neval": state["n"], "loss": loss, "gnorm": float(g.norm())})
        if verbose and state["n"] % 10 == 0:
            print(f"[lbfgs {state['n']:4d}] loss={loss:.4f} ||g||={float(g.norm()):.4e}")
        return float(loss), g.double().cpu().numpy().astype(np.float64)

    x0 = theta0.reshape(-1).double().cpu().numpy().astype(np.float64)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-12, "gtol": 1e-8})
    theta = torch.tensor(res.x, device=dev, dtype=dt).reshape(S, 3)
    return theta, state["hist"]
