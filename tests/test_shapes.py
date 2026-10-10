import pytest
import torch

from src.baselines.resnet_small import ResNetSmall
from src.models.attention import Attention
from src.models.block import Block
from src.models.droppath import DropPath
from src.models.mlp import MLP
from src.models.patch_embed import PatchEmbedding
from src.models.pos_encoding import PositionalEncoding, sinusoidal
from src.models.vit import VisionTransformer


@pytest.mark.parametrize("method", ["conv", "unfold"])
def test_patch_shape(method):
    assert PatchEmbedding(method=method)(torch.randn(2, 3, 32, 32)).shape == (2, 64, 64)


@pytest.mark.parametrize("kind", ["learned", "sinusoidal", "sincos2d", "none"])
@pytest.mark.parametrize("cls", [True, False])
def test_position_shape(kind, cls):
    module = PositionalEncoding(4, 32, kind, cls)
    x = torch.zeros(2, 16 + int(cls), 32)
    assert module(x).shape == x.shape
    assert len(list(module.parameters())) == int(kind == "learned")
    if kind == "none":
        torch.testing.assert_close(module(x), x)


def test_position_values():
    table = sinusoidal(3, 5)
    torch.testing.assert_close(table[0, 0::2], torch.zeros(3))
    torch.testing.assert_close(table[0, 1::2], torch.ones(2))
    module = PositionalEncoding(2, 8, "sincos2d", False)
    # First channel half describes rows; second half describes columns.
    torch.testing.assert_close(module.encoding[0, 0, :4], module.encoding[0, 1, :4])
    torch.testing.assert_close(module.encoding[0, 0, 4:], module.encoding[0, 2, 4:])
    assert not torch.equal(module.encoding[0, 0], module.encoding[0, 3])


def test_modules():
    x = torch.randn(2, 17, 32)
    output, weights = Attention(32, 4)(x, True)
    assert output.shape == x.shape and weights.shape == (2, 4, 17, 17)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 17))
    assert MLP(32, 128)(x).shape == x.shape
    assert Block(32, 4)(x).shape == x.shape
    assert DropPath(0.5).eval()(x) is x
    assert ResNetSmall()(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_droppath_mask_and_expectation():
    x = torch.ones(10000, 3, 4)
    output = DropPath(0.5)(x)
    assert set(output.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(output[:, 0, 0], output[:, 2, 3])
    assert abs(float(output.mean()) - 1) < 0.04


@pytest.mark.parametrize("pooling", ["cls", "avg"])
def test_vit_shape_and_maps(pooling):
    model = VisionTransformer(embed_dim=32, depth=2, pooling=pooling)
    logits, maps = model(torch.randn(2, 3, 32, 32), True)
    assert logits.shape == (2, 10)
    assert len(maps) == 2
    n = 64 + int(pooling == "cls")
    assert maps[0].shape == (2, 4, n, n)


def test_invalid_geometry():
    with pytest.raises(ValueError):
        PatchEmbedding(patch_size=7)
    with pytest.raises(ValueError):
        PatchEmbedding()(torch.randn(2, 3, 33, 32))
    with pytest.raises(ValueError):
        Attention(30, 4)
    with pytest.raises(ValueError):
        PositionalEncoding(4, 6, "sincos2d")
