"""Complete from-scratch Vision Transformer. No timm or fused-attention imports."""

import torch
from torch import Tensor, nn

from .block import Block
from .patch_embed import PatchEmbedding
from .pos_encoding import PositionalEncoding


class VisionTransformer(nn.Module):
    """[B,C,H,W] -> logits [B,K]. CLS or mean of normalized patch tokens.

    With pooling='avg' no unused CLS parameter exists. Position encodings
    include exactly the tokens the model uses. Final normalization occurs
    before pooling, matching timm when fc_norm=False.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        positional_encoding: str = "learned",
        pooling: str = "cls",
        patch_method: str = "conv",
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "avg"} or depth <= 0 or num_classes <= 0 or mlp_ratio <= 0:
            raise ValueError("Invalid pooling, depth, class count, or MLP ratio")
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0,1)")
        self.pooling = pooling
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim, patch_method
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pooling == "cls" else None
        self.pos = PositionalEncoding(
            image_size // patch_size, embed_dim, positional_encoding, pooling == "cls"
        )
        self.pos_drop = nn.Dropout(dropout)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, rate, norm_eps)
                for rate in rates
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._initialize)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """[B,C,H,W] -> [B,N,D] -> [B,D] -> [B,K]; optional weights per layer."""
        tokens = self.patch_embed(x)
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), tokens], dim=1)
        tokens = self.pos_drop(self.pos(tokens))
        maps = []
        for block in self.blocks:
            if return_attention:
                tokens, weights = block(tokens, True)
                maps.append(weights)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        features = tokens[:, 0] if self.pooling == "cls" else tokens.mean(dim=1)
        logits = self.head(features)
        return (logits, maps) if return_attention else logits
