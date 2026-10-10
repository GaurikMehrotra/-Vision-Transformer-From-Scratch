import torch

from src.models.vit import VisionTransformer
from src.utils import seed_everything


def test_initialization_and_training_dropout_deterministic_with_seed():
    images = torch.randn(2, 3, 32, 32)
    seed_everything(7)
    a = VisionTransformer(embed_dim=32, depth=2, dropout=0.2, drop_path_rate=0.2)
    first = a(images)
    seed_everything(7)
    b = VisionTransformer(embed_dim=32, depth=2, dropout=0.2, drop_path_rate=0.2)
    torch.testing.assert_close(first, b(images), atol=0, rtol=0)
    a.eval()
    torch.testing.assert_close(a(images), a(images), atol=0, rtol=0)
