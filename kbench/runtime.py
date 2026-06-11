"""Drive the vendored forward / backward passes from captured tensors.

This module reconstructs the minimal ``static`` object the solver expects (out of a captured dict
of tensors) and exposes two entry points:

  * :func:`run_forward`  -- theta -> (loss, forward intermediates)
  * :func:`run_backward` -- (forward intermediates) -> (grad_theta, grad_col)

Both call the SAME vendored code paths the bench measures; the only editable surface underneath them
is ``kbench/core/kernels/*``. Nothing here should need editing by the kernel optimizer.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from kbench.api.solver_options import SolverOptions
from kbench.api._implicit_grad import implicit_grad_loglik_vjp_wave
from kbench.core.inference.solver import (
    nll_from_root_rows,
    col_weights_are_uniform,
    solve_e_pi,
)

# Names of the tensors the forward produces and the backward consumes, in save order.
# Kept here so capture + bench agree.
FORWARD_SAVED_NAMES = (
    "E", "E_s1", "E_s2", "Ebar", "root_rows", "pi_wave", "pibar_wave",
    "pibar_row_max", "log_pS", "log_pD", "log_pL", "max_coupling", "col_log_probs",
)


def move_to_device(obj, device):
    """Recursively move every tensor inside a nested dict/list to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def make_static(captured: dict, device) -> SimpleNamespace:
    """Rebuild the ``static`` object solve_e_pi / implicit_grad expect."""
    st = captured["static"]
    solver_options = SolverOptions(**captured["meta"]["solver_options"])
    solver_options.validate()
    return SimpleNamespace(
        wave_layout=move_to_device(st["wave_layout"], device),
        state_helpers=move_to_device(st["state_helpers"], device),
        rate_item_idx=move_to_device(st["rate_item_idx"], device),
        itemwise=bool(st["itemwise"]),
        statewise=bool(st["statewise"]),
        solver_options=solver_options,
        warm_E=None,
    )


def run_forward(static, theta: torch.Tensor, col_weights: torch.Tensor):
    """theta -> (loss, dict of forward intermediates). Mirrors the reference forward."""
    with torch.no_grad():
        out = solve_e_pi(static, theta, col_weights, warm_start_E=None)
        (E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max,
         log_pS, log_pD, log_pL, max_coupling, col_log_probs) = out
        loss = nll_from_root_rows(root_rows, E)
    saved = dict(zip(FORWARD_SAVED_NAMES, (
        E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max,
        log_pS, log_pD, log_pL, max_coupling, col_log_probs,
    )))
    return loss, saved


def run_backward(static, theta, col_weights, saved: dict):
    """(forward intermediates) -> (grad_theta, grad_col). Mirrors the reference backward.

    ``saved`` provides the frozen forward intermediates so the backward is timed and
    checked independently of the forward (the golden forward outputs are its input).
    """
    so = static.solver_options
    use_col_weights = not col_weights_are_uniform(col_weights)
    grad_theta, grad_col = implicit_grad_loglik_vjp_wave(
        static.wave_layout,
        static.state_helpers,
        Pi_star_wave=saved["pi_wave"],
        Pibar_star_wave=saved["pibar_wave"],
        E_star=saved["E"],
        Ebar=saved["Ebar"],
        E_s1=saved["E_s1"],
        E_s2=saved["E_s2"],
        log_pS=saved["log_pS"],
        log_pD=saved["log_pD"],
        log_pL=saved["log_pL"],
        max_coupling_mat=saved["max_coupling"],
        col_log_probs=saved["col_log_probs"],
        use_col_weights=use_col_weights,
        theta=theta,
        col_weights=col_weights,
        item_idx=static.rate_item_idx,
        uniform_pibar_row_max=saved["pibar_row_max"],
        statewise=static.statewise,
        itemwise=static.itemwise,
        neumann_terms=so.neumann_terms,
        self_loop_solver=so.self_loop_solver,
        bicgstab_max_iter=so.bicgstab_max_iter,
        bicgstab_tol=so.bicgstab_tol,
        bicgstab_breakdown_tol=so.bicgstab_breakdown_tol,
        adjoint_pruning_threshold=so.adjoint_pruning_threshold,
        use_adjoint_pruning=so.use_adjoint_pruning,
        pibar_side_threshold=so.pibar_side_threshold,
    )
    return grad_theta, grad_col
