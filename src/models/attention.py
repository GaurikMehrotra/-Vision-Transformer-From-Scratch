"""Multi-head self-attention implemented with projections and matrix products."""

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    """[B,N,D] -> [B,N,D]; optionally also return probabilities [B,H,N,N].

    Queries and keys have head dimension Dh=D/H. Dividing QK^T by sqrt(Dh)
    keeps scores from growing with Dh. Softmax normalizes over key tokens.
    Returned weights are before dropout, so rows sum to one in training too.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if min(embed_dim, num_heads) <= 0 or embed_dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by positive head count")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Split heads, attend over N, then merge H*Dh back into D."""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # [B,H,N,Dh] @ [B,H,Dh,N] -> [B,H,N,N].
        probabilities = (torch.matmul(q, k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        values = torch.matmul(self.attn_drop(probabilities), v)
        values = values.transpose(1, 2).reshape(b, n, d)
        output = self.proj_drop(self.proj(values))
        return (output, probabilities) if return_attention else output
