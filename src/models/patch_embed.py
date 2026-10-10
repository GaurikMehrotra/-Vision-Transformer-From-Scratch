"""Image [B,C,H,W] to patch tokens [B,(H/P)*(W/P),D]."""

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Non-overlapping patches in row-major order, using conv or unfold+linear.

    Conv weights [D,C,P,P] flatten to linear weights [D,C*P*P]. The
    flattened order is channel, patch row, patch column for both methods.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 64,
        method: str = "conv",
    ) -> None:
        super().__init__()
        if min(image_size, patch_size, in_channels, embed_dim) <= 0 or image_size % patch_size:
            raise ValueError("Positive dimensions and image_size divisible by patch_size required")
        if method not in {"conv", "unfold"}:
            raise ValueError("method must be conv or unfold")
        self.image_size, self.patch_size, self.in_channels = image_size, patch_size, in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.method = method
        self.proj = (
            nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
            if method == "conv"
            else nn.Linear(in_channels * patch_size**2, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """[B,C,H,W] -> [B,N,D]; reject unexpected geometry rather than truncate."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (
            self.in_channels,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("Expected [B,C,image_size,image_size]")
        if self.method == "conv":
            return self.proj(x).flatten(2).transpose(1, 2)
        # Unfold appends patch height/width; move C behind patch-grid dimensions.
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], self.num_patches, -1)
        return self.proj(patches)
