import os
from numbers import Integral, Real

import torch

GIB = 1024 ** 3


def _int_dimension(name: str, value: int | float) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number)
    raise ValueError(f"{name} must be an integer")


def _nonnegative_int(name: str, value: int | float) -> int:
    number = _int_dimension(name, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_int(name: str, value: int | float) -> int:
    number = _int_dimension(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def cuda_memory_budget_bytes(
    device: torch.device | int | None = None,
    *,
    default_fraction: float = 0.85,
    default_reserve_gib: float = 1.0,
) -> int | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        idx = int(device)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    fraction = float(os.environ.get("GPUREC_MEMORY_POLICY_FRACTION", str(default_fraction)))
    reserve_b = int(float(os.environ.get("GPUREC_MEMORY_POLICY_RESERVE_GIB", str(default_reserve_gib))) * GIB)
    return max(0, min(int(total_b * fraction), max(0, int(free_b) - reserve_b)))


def proposal0_wave_scratch_bytes(W: int, S: int, dtype: torch.dtype, *, scratch_tensors: int = 10) -> int:
    return (
        _nonnegative_int("W", W)
        * _positive_int("S", S)
        * _nonnegative_int("scratch_tensors", scratch_tensors)
        * dtype_nbytes(dtype)
    )


def proposal0_memory_gate(
    W: int,
    S: int,
    dtype: torch.dtype,
    *,
    device: torch.device | int | None = None,
    already_live_bytes: int = 0,
) -> tuple[bool, int, int | None]:
    required = _nonnegative_int("already_live_bytes", already_live_bytes) + proposal0_wave_scratch_bytes(W, S, dtype)
    budget = cuda_memory_budget_bytes(device)
    if budget is None:
        return True, required, budget
    return required <= budget, required, budget
