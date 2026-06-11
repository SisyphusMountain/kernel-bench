import torch

_NEG_INF = float("-inf")


def safe_log2(x: torch.Tensor) -> torch.Tensor:
    positive = x > 0
    return torch.where(positive, torch.log2(torch.where(positive, x, torch.ones_like(x))), x.new_full(x.shape, _NEG_INF))


def logsumexp2(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    row_max = x.max(dim=dim, keepdim=True).values
    row_max_safe = torch.where(row_max == _NEG_INF, torch.zeros_like(row_max), row_max)
    total = torch.exp2(x - row_max_safe).sum(dim=dim, keepdim=True)
    result = safe_log2(total) + row_max
    return result if keepdim else result.squeeze(dim)
