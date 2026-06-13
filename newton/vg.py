"""Value-and-gradient closure over the frozen forward/backward, plus fixture loading.

``make_value_and_grad`` wraps ``solve_e_pi`` (with ``warm_start_E`` support, which ``run_forward``
does not expose) + ``run_backward`` into a single callable the Newton-CG loop drives. The
optimization variable is the flat ``theta`` vector (``theta[S,3].reshape(-1)``); the gradient layout
is elementwise-identical, so flatten/unflatten is a plain ``reshape``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))  # make `kbench` importable without install

from kbench.runtime import FORWARD_SAVED_NAMES, make_static, move_to_device, run_backward  # noqa: E402
from kbench.core.inference.solver import nll_from_root_rows, solve_e_pi  # noqa: E402

DATA = _REPO / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def free_cuda_cache_if_tight(min_free_gib: float = 4.0):
    """Release the caching allocator's pool to the driver when driver-free memory runs low.

    kbench's backward gates its scratch on ``torch.cuda.mem_get_info`` (driver-free bytes), which
    the caching allocator does not replenish on tensor free — without this, long optimization
    loops on the big fixtures trip the gate spuriously.
    """
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b < min_free_gib * (1024 ** 3):
            torch.cuda.empty_cache()


def load_problem(label: str = "small", device: str = DEVICE):
    """Load a captured fixture and rebuild (cap, static, theta, col_weights) on ``device``."""
    path = DATA / label / "whole.pt"
    if not path.exists():
        raise FileNotFoundError(f"no capture at {path} -- run capture/capture.py first")
    cap = torch.load(path, map_location="cpu", weights_only=False)
    static = make_static(cap, device)
    theta = move_to_device(cap["inputs"]["theta"], device).contiguous()
    col_weights = move_to_device(cap["inputs"]["col_weights"], device).contiguous()
    return cap, static, theta, col_weights


def forward_solve(static, theta: torch.Tensor, col_weights: torch.Tensor, *, warm_E=None):
    """Run the forward solve at ``theta``; return (loss_tensor, saved_dict). Mirrors run_forward."""
    with torch.no_grad():
        out = solve_e_pi(static, theta, col_weights, warm_start_E=warm_E)
        saved = dict(zip(FORWARD_SAVED_NAMES, out))
        loss = nll_from_root_rows(saved["root_rows"], saved["E"])
    return loss, saved


def make_value_and_grad(static, col_weights: torch.Tensor, *, grad_avg_K: int = 1):
    """Return ``f(theta_vec, *, warm_E, want_grad) -> (loss, g_vec, saved, warm_E_out)``.

    ``loss`` is a Python float, ``g_vec`` a length-3S fp32 tensor (or None when want_grad=False),
    ``saved`` the forward intermediates run_backward consumed, ``warm_E_out`` = saved['E'] for
    warm-starting the next nearby solve. ``grad_avg_K`` averages the (atomically nondeterministic)
    backward to suppress its noise floor in the HVP/CG path.
    """
    S = int(static.state_helpers["S"])

    def f(theta_vec: torch.Tensor, *, warm_E=None, want_grad: bool = True):
        theta = theta_vec.detach().reshape(S, 3).contiguous()
        loss_t, saved = forward_solve(static, theta, col_weights, warm_E=warm_E)
        loss = float(loss_t)
        g_vec = None
        if want_grad:
            # the backward's scratch gate reads driver-free memory; return any stale cached
            # blocks (e.g. from another dtype's stage) before it runs
            free_cuda_cache_if_tight()
            acc = None
            for _ in range(int(grad_avg_K)):
                gt, _gc = run_backward(static, theta, col_weights, saved)
                acc = gt if acc is None else acc + gt
            g_vec = (acc / float(grad_avg_K)).reshape(-1).contiguous()
        return loss, g_vec, saved, saved["E"]

    return f


def _smoke_test(label: str = "small") -> bool:
    """Reproduce golden loss + grad_theta at the fixture theta within 2e-3."""
    cap, static, theta, col_weights = load_problem(label)
    S = int(static.state_helpers["S"])
    f = make_value_and_grad(static, col_weights)
    theta_vec = theta.reshape(-1).clone()
    loss, g_vec, saved, warm_E = f(theta_vec)

    gold = cap["golden"]
    gold_loss = float(gold["loss"])
    gold_g = gold["grad_theta"].to(g_vec.device).reshape(-1).float()
    g = g_vec.float()

    loss_abs = abs(loss - gold_loss)
    g_abs = float((g - gold_g).abs().max())
    g_rel = float(((g - gold_g).abs() / gold_g.abs().clamp_min(1e-12)).max())
    g_norm = float(g.norm())
    ok = loss_abs <= max(2e-3, 2e-3 * abs(gold_loss)) and (g_abs <= 2e-3 or g_rel <= 2e-3)
    print(f"[vg smoke {label}] S={S} p={3*S}")
    print(f"  loss        got={loss:.6f} gold={gold_loss:.6f} abs={loss_abs:.3e}")
    print(f"  grad_theta  max_abs={g_abs:.3e} max_rel={g_rel:.3e}  ||g||2={g_norm:.4f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys as _sys

    label = _sys.argv[1] if len(_sys.argv) > 1 else "small"
    raise SystemExit(0 if _smoke_test(label) else 1)
