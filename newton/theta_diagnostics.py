"""Part of the specieswise basin investigation (2026-06-15); see
newton/_specieswise_basin_findings.md for context. Run from the repo root with
`python -m newton.theta_diagnostics` (pure CPU, no CUDA required).

Is the 137385 basin a genuine interior minimum or a boundary/saturated solution? Characterize the
theta distributions + the implied 4-category probabilities for the deep basin vs the old basin vs the
fixture init. theta columns are base-2 logits: p = softmax([0, theta_S, theta_D, theta_L] * ln2)."""
import math
import os
import torch

CKPT_DIR = os.path.join(os.path.dirname(__file__), "_checkpoints")
LN2 = math.log(2.0)
S = 1331


def probs(theta):  # theta [S,3] -> p [S,4] = [ref, pS, pD, pL]
    logits = torch.cat([torch.zeros(theta.shape[0], 1), theta], dim=1)
    return torch.softmax(logits * LN2, dim=1)


def describe(name, theta):
    theta = theta.reshape(S, 3).float().cpu()
    p = probs(theta)
    pmax = p.max(dim=1).values            # most-likely category prob per row
    dtl = p[:, 1:]                        # the 3 DTL probs per row
    th = theta.reshape(-1)
    q = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
    tq = torch.quantile(th, q)
    print(f"\n=== {name} ===")
    print(f"  theta: min={th.min():.2f} 1%={tq[1]:.2f} median={tq[2]:.2f} 99%={tq[3]:.2f} max={th.max():.2f}")
    print(f"  |theta|>5: {(th.abs()>5).float().mean()*100:.1f}%   |theta|>10: {(th.abs()>10).float().mean()*100:.1f}%"
          f"   |theta|>15: {(th.abs()>15).float().mean()*100:.1f}%")
    print(f"  rows with max-prob > 0.90: {(pmax>0.90).float().mean()*100:.1f}%   > 0.99: {(pmax>0.99).float().mean()*100:.1f}%"
          f"   > 0.999: {(pmax>0.999).float().mean()*100:.1f}%")
    print(f"  DTL probs < 1e-3: {(dtl<1e-3).float().mean()*100:.1f}%   < 1e-5: {(dtl<1e-5).float().mean()*100:.1f}%")
    print(f"  median DTL prob = {dtl.median():.4f}   mean ref(no-event) prob = {p[:,0].mean():.4f}")
    # which column saturates? count rows where each DTL column is the argmax & >0.9
    for j, nm in enumerate(["ref", "pS", "pD", "pL"]):
        sat = ((p.argmax(1) == j) & (pmax > 0.9)).float().mean() * 100
        print(f"    rows saturated (>0.9) on {nm}: {sat:.1f}%")


fixture = torch.full((S, 3), math.log2(0.1))   # the fixture theta0 (-3.32193 = log2(0.1))
describe("fixture init (theta0)", fixture)
for path, nm in [(os.path.join(CKPT_DIR, "old_basin_137466.pt"), "OLD basin 137466"),
                 (os.path.join(CKPT_DIR, "specieswise_best_137384.pt"), "DEEP basin 137385.88")]:
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        describe(nm, d["theta"])
    except Exception as e:
        print(f"\n{nm}: load error {e}")
