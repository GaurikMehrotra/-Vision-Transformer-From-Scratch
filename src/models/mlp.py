"""Token-wise feed-forward network: [B,N,D] -> [B,N,D]."""

from torch import Tensor, nn


class MLP(nn.Module):
    """Expand each token D -> hidden_dim, apply GELU, then project back to D."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """[B,N,D] -> [B,N,hidden_dim] -> [B,N,D]."""
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
