"""Pre-LayerNorm Transformer block, preserving token shape [B,N,D]."""

from torch import Tensor, nn

from .attention import Attention
from .droppath import DropPath
from .mlp import MLP


class Block(nn.Module):
    """x <- x + Attention(LN(x)); x <- x + MLP(LN(x))."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = Attention(embed_dim, num_heads, attn_dropout, dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """[B,N,D] -> [B,N,D], optionally returning [B,H,N,N] attention."""
        result = self.attn(self.norm1(x), return_attention)
        if return_attention:
            branch, weights = result
        else:
            branch = result
        x = x + self.drop_path1(branch)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return (x, weights) if return_attention else x
