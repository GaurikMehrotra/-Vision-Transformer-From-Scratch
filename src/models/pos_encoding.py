"""Position information for tokens [B,N,D], with optional CLS prefix."""

import math

import torch
from torch import Tensor, nn


def sinusoidal(length: int, dim: int) -> Tensor:
    """[length,dim] table: even dimensions are sine, odd dimensions cosine."""
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(positions * frequencies)
    table[:, 1::2] = torch.cos(positions * frequencies[: dim // 2])
    return table


class PositionalEncoding(nn.Module):
    """Add [1,N,D] encoding to [B,N,D]; four selectable position strategies.

    Fixed 2D uses half the channels for each grid axis. The CLS position is
    zero for 2D encoding because it has no image coordinate. Fixed encodings
    are buffers: checkpointed and moved with the module, never optimized.
    """

    def __init__(
        self, grid_size: int, embed_dim: int, kind: str = "learned", cls_token: bool = True
    ) -> None:
        super().__init__()
        if min(grid_size, embed_dim) <= 0:
            raise ValueError("Positive grid and embedding dimensions required")
        n = grid_size**2 + int(cls_token)
        self.kind = kind
        if kind == "learned":
            self.encoding = nn.Parameter(torch.empty(1, n, embed_dim))
            nn.init.trunc_normal_(self.encoding, std=0.02)
        elif kind == "sinusoidal":
            self.register_buffer("encoding", sinusoidal(n, embed_dim)[None])
        elif kind == "sincos2d":
            if embed_dim % 4:
                raise ValueError("2D sin-cos requires embedding dimension divisible by 4")
            axis = sinusoidal(grid_size, embed_dim // 2)
            rows = axis[:, None, :].expand(grid_size, grid_size, -1)
            cols = axis[None, :, :].expand(grid_size, grid_size, -1)
            table = torch.cat([rows, cols], dim=-1).reshape(grid_size**2, embed_dim)
            if cls_token:
                table = torch.cat([torch.zeros(1, embed_dim), table])
            self.register_buffer("encoding", table[None])
        elif kind == "none":
            self.register_buffer("encoding", torch.zeros(1, n, embed_dim))
        else:
            raise ValueError(f"Unknown positional encoding: {kind}")

    def forward(self, x: Tensor) -> Tensor:
        """[B,N,D] -> [B,N,D], broadcasting the same positions across images."""
        if x.ndim != 3 or x.shape[1:] != self.encoding.shape[1:]:
            raise ValueError("Token geometry differs from positional encoding")
        return x + self.encoding.to(dtype=x.dtype)
