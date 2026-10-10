"""Stochastic depth drops an entire residual branch independently per image."""

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    """[B,...] -> [B,...]; mask shape [B,1,...] broadcasts over each sample."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("DropPath probability must be in [0,1)")
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        """Scale surviving branches by 1/(1-p) to preserve their expectation."""
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep
        return x * mask.to(x.dtype) / keep
