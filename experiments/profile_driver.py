"""NVTX-annotated driver for nsys/ncu profiling of the whole forward/backward pass.

Loads one captured size, warms up, then runs N annotated iterations of forward and/or
backward.  Use with:

    nsys profile -o out --capture-range=cudaProfilerApi \
        python experiments/profile_driver.py --size 1007x64 --iters 3

    ncu --replay-mode kernel ... python experiments/profile_driver.py --size 1007x64 \
        --iters 1 --pass backward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))

from _common import DEVICE, load_whole  # noqa: E402
from kbench.runtime import make_static, move_to_device, run_backward, run_forward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="1007x64")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pass", dest="which", choices=["forward", "backward", "both"], default="both")
    args = ap.parse_args()

    cap = load_whole(args.size)
    static = make_static(cap, DEVICE)
    theta = move_to_device(cap["inputs"]["theta"], DEVICE)
    rw = move_to_device(cap["inputs"]["col_weights"], DEVICE)
    saved = move_to_device(cap["forward_saved"], DEVICE)
    m = cap["meta"]
    print(f"profiling [{args.size}] S={m['S']} C={m['C']} items={m['n_items']} pass={args.which}")

    do_fwd = args.which in ("forward", "both")
    do_bwd = args.which in ("backward", "both")

    for _ in range(args.warmup):
        if do_fwd:
            run_forward(static, theta, rw)
        if do_bwd:
            run_backward(static, theta, rw, saved)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iters):
        if do_fwd:
            torch.cuda.nvtx.range_push(f"forward_{i}")
            run_forward(static, theta, rw)
            torch.cuda.nvtx.range_pop()
        if do_bwd:
            torch.cuda.nvtx.range_push(f"backward_{i}")
            run_backward(static, theta, rw, saved)
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("done")


if __name__ == "__main__":
    main()
