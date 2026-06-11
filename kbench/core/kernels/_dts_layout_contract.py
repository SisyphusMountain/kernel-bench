"""Shared DTS parameter layout contract for private kernel wrappers.

The retained forward and backward DTS wrappers historically resolve the
item-indexed one-dimensional case differently:

* forward treats a 1-D tensor of length ``S`` as a shared state vector;
* backward treats any 1-D tensor as item scalar rows when ``item_idx`` is
  present.

This module keeps that precedence explicit and CPU-testable.  It only
classifies layout intent; callers still own tensor normalization and the exact
Triton launch arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Any, Sequence


class DtsForwardAddressMode(IntEnum):
    """Addressing modes consumed by ``dts_fused`` Triton kernels."""

    SHARED_STATE = 0
    ITEM_INDEXED = 4


class DtsBackwardLayoutCode(IntEnum):
    """Parameter layout codes consumed by the retained DTS backward kernel."""

    SHARED_SCALAR = 0
    SHARED_STATE = 1
    ITEM_SCALAR = 2
    ITEM_STATE = 3


@dataclass(frozen=True)
class DtsForwardParamLayout:
    """Forward parameter addressing metadata derived from tensor layout."""

    mode: DtsForwardAddressMode
    row_stride: int
    state_stride: int
    intent: str
    ambiguous_1d_item_state: bool = False


@dataclass(frozen=True)
class DtsBackwardParamLayout:
    """Backward parameter layout code plus documented shape intent."""

    code: DtsBackwardLayoutCode
    intent: str
    ambiguous_1d_item_state: bool = False


def dts_forward_param_layout(
    value_or_shape: Any,
    *,
    S: int,
    item_indexed: bool,
    strides: Sequence[int] | None = None,
) -> DtsForwardParamLayout:
    """Classify the direct DTS forward parameter layout.

    Item-indexed forward keeps the existing precedence where a 1-D tensor of
    length ``S`` is shared state-indexed.  When item count also equals
    ``S``, direct callers must use ``[G, 1]`` for item scalar rows to avoid
    disagreement with retained backward.
    """

    state_count = _positive_count("S", S)
    shape = _shape_tuple(value_or_shape)
    stride = _stride_tuple(value_or_shape, shape=shape, strides=strides)

    if item_indexed:
        if len(shape) == 0:
            return DtsForwardParamLayout(
                DtsForwardAddressMode.ITEM_INDEXED,
                row_stride=0,
                state_stride=0,
                intent="shared_scalar",
            )
        if len(shape) == 1:
            if _numel(shape) == state_count:
                return DtsForwardParamLayout(
                    DtsForwardAddressMode.ITEM_INDEXED,
                    row_stride=0,
                    state_stride=int(stride[0]),
                    intent="shared_state",
                    ambiguous_1d_item_state=True,
                )
            return DtsForwardParamLayout(
                DtsForwardAddressMode.ITEM_INDEXED,
                row_stride=int(stride[0]),
                state_stride=0,
                intent="item_scalar",
            )
        if len(shape) == 2:
            if shape[1] == 1:
                return DtsForwardParamLayout(
                    DtsForwardAddressMode.ITEM_INDEXED,
                    row_stride=int(stride[0]),
                    state_stride=0,
                    intent="item_scalar",
                )
            if shape[1] == state_count:
                return DtsForwardParamLayout(
                    DtsForwardAddressMode.ITEM_INDEXED,
                    row_stride=0 if shape[0] == 1 else int(stride[0]),
                    state_stride=int(stride[1]),
                    intent="item_state",
                )
        raise ValueError(
            "item-indexed DTS parameters must be scalar, [S], [G], [G, 1], "
            f"or [G, S]; got shape {shape} with S={state_count}"
        )

    if len(shape) == 0:
        return DtsForwardParamLayout(
            DtsForwardAddressMode.SHARED_STATE,
            row_stride=0,
            state_stride=1,
            intent="shared_scalar_expanded_to_state",
        )
    if len(shape) == 1 and _numel(shape) == state_count:
        return DtsForwardParamLayout(
            DtsForwardAddressMode.SHARED_STATE,
            row_stride=0,
            state_stride=1,
            intent="shared_state",
        )
    raise ValueError(
        "DTS parameters must be scalar or [S] without item indexing; "
        f"got shape {shape} with S={state_count}"
    )


def dts_backward_param_layout(
    value_or_shape: Any,
    *,
    S: int,
    item_indexed: bool,
) -> DtsBackwardParamLayout:
    """Classify the retained DTS backward parameter or gradient layout.

    With ``item_indexed`` true, retained backward intentionally classifies any
    1-D non-scalar tensor as item scalar rows before considering ``[S]``.
    This preserves current kernel semantics and documents why bare ``[G]`` is
    ambiguous when ``G == S``.
    """

    state_count = _positive_count("S", S)
    shape = _shape_tuple(value_or_shape)

    if _numel(shape) == 1:
        return DtsBackwardParamLayout(
            DtsBackwardLayoutCode.SHARED_SCALAR,
            intent="shared_scalar",
        )
    if item_indexed and len(shape) == 1:
        return DtsBackwardParamLayout(
            DtsBackwardLayoutCode.ITEM_SCALAR,
            intent="item_scalar",
            ambiguous_1d_item_state=shape[0] == state_count,
        )
    if len(shape) == 1 and shape[0] == state_count:
        return DtsBackwardParamLayout(
            DtsBackwardLayoutCode.SHARED_STATE,
            intent="shared_state",
        )
    if item_indexed:
        if len(shape) == 2 and shape[1] == 1:
            return DtsBackwardParamLayout(
                DtsBackwardLayoutCode.ITEM_SCALAR,
                intent="item_scalar",
            )
        if len(shape) == 2 and shape[1] == state_count:
            return DtsBackwardParamLayout(
                DtsBackwardLayoutCode.ITEM_STATE,
                intent="item_state",
            )
    raise ValueError(
        "DTS parameters must be scalar, [S], [G], [G, 1], or [G, S]; "
        f"got shape {shape} with S={state_count}"
    )


def _shape_tuple(value_or_shape: Any) -> tuple[int, ...]:
    raw_shape = getattr(value_or_shape, "shape", value_or_shape)
    if isinstance(raw_shape, (str, bytes)):
        raise ValueError("DTS parameter shape must be a sequence of dimensions")
    try:
        dims = tuple(raw_shape)
    except TypeError as exc:
        raise ValueError("DTS parameter shape must be a sequence of dimensions") from exc

    shape: list[int] = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            raise ValueError("DTS parameter shape dimensions must be integers")
        size = int(dim)
        if size < 0:
            raise ValueError("DTS parameter shape dimensions must be non-negative")
        shape.append(size)
    return tuple(shape)


def _stride_tuple(
    value_or_shape: Any,
    *,
    shape: tuple[int, ...],
    strides: Sequence[int] | None,
) -> tuple[int, ...]:
    raw_strides = strides
    stride_method = getattr(value_or_shape, "stride", None)
    if raw_strides is None and callable(stride_method):
        raw_strides = stride_method()
    if raw_strides is None:
        return _contiguous_strides(shape)
    try:
        stride = tuple(raw_strides)
    except TypeError as exc:
        raise ValueError("DTS parameter strides must be a sequence of integers") from exc
    if len(stride) != len(shape):
        raise ValueError("DTS parameter strides must match shape rank")
    values: list[int] = []
    for item in stride:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError("DTS parameter strides must be integers")
        values.append(int(item))
    return tuple(values)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides: list[int] = []
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(1, size)
    return tuple(reversed(strides))


def _positive_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


__all__ = [
    "DtsBackwardLayoutCode",
    "DtsBackwardParamLayout",
    "DtsForwardAddressMode",
    "DtsForwardParamLayout",
    "dts_backward_param_layout",
    "dts_forward_param_layout",
]
