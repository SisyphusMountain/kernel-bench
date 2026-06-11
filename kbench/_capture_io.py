"""Capture/replay helpers shared by capture/ and bench/ (no triton import here).

Kernel wrappers are called with tensors that *alias* each other (the forward wave loop passes
the same ``pibar`` buffer as both the output and the Pibar argument on even iterations) and that
they mutate in place. They also receive large ``[C, S]`` buffers (Pi, Pibar) that recur, identical,
across many captured calls. So a capture is stored in two layers:

  * a per-size **content-addressed pool**: every distinct tensor (by bytes) is stored once;
  * per-call **ref structures**: the (args, kwargs[, ret]) structure with each tensor replaced by a
    ``Ref(content_idx, group_id)`` -- ``content_idx`` points into the pool, ``group_id`` records
    which arguments shared storage in the original call.

``rebuild`` reconstructs a call: one fresh tensor per ``group_id`` (so in-place writes alias exactly
as in the original, and distinct-but-equal arguments stay distinct), materialized lazily from the
pool -- only referenced entries are touched, and passing ``device`` moves just those.
"""

from __future__ import annotations

import hashlib

import torch


class Ref:
    """Picklable sentinel: (pool content index, per-call storage-aliasing group id)."""

    __slots__ = ("c", "g")

    def __init__(self, c: int, g: int):
        self.c = int(c)
        self.g = int(g)

    def __repr__(self):
        return f"Ref(c={self.c}, g={self.g})"


def content_key(t: torch.Tensor):
    """Identity by (dtype, shape, content hash) so equal buffers dedup to one pool entry."""
    t = t.detach().contiguous().cpu()
    digest = hashlib.blake2b(t.view(torch.uint8).numpy().tobytes(), digest_size=16).hexdigest()
    return (str(t.dtype), tuple(t.shape), digest)


def dedup_into(struct, pool: list, key2idx: dict):
    """Rewrite a nested struct into Refs, appending newly-seen tensors to ``pool``.

    ``group_id`` is assigned per distinct storage *within this struct*, so aliased arguments
    (same storage) get the same group and are re-shared on rebuild.
    """
    grp = {}  # storage data_ptr -> group id (local to this struct)

    def conv(o):
        if torch.is_tensor(o):
            g = grp.setdefault(o.untyped_storage().data_ptr(), len(grp))
            key = content_key(o)
            if key not in key2idx:
                key2idx[key] = len(pool)
                pool.append(o.detach().clone().cpu())
            return Ref(key2idx[key], g)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(conv(v) for v in o)
        return o

    return conv(struct)


def rebuild(ref_struct, pool, *, device=None):
    """Inverse of dedup_into. One tensor per group_id (aliasing preserved); lazy from pool.

    With ``device`` set, each group's tensor is moved to that device (a fresh copy, so the pool
    stays clean and groups are isolated). With ``device=None`` the pool's CPU tensors are returned
    directly -- fine for read-only golden comparison.
    """
    cache = {}  # group id -> materialized tensor

    def conv(o):
        if isinstance(o, Ref):
            if o.g not in cache:
                t = pool[o.c]
                cache[o.g] = t.to(device) if device is not None else t
            return cache[o.g]
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(conv(v) for v in o)
        return o

    return conv(ref_struct)


def flatten_tensors(obj, prefix: str, out: list):
    """Append (name, tensor) for every tensor leaf, in a stable order."""
    if torch.is_tensor(obj):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten_tensors(v, f"{prefix}.{k}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten_tensors(v, f"{prefix}[{i}]", out)
