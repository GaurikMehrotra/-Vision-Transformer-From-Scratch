import pytest
import torch

from src.models.vit import VisionTransformer
from src.utils import load_config, optimizer_for


@pytest.mark.parametrize("pooling", ["cls", "avg"])
@pytest.mark.parametrize("positions", ["learned", "sinusoidal", "sincos2d", "none"])
@pytest.mark.parametrize("patch", ["conv", "unfold"])
def test_all_parameters_receive_finite_gradients(pooling, positions, patch):
    model = VisionTransformer(
        embed_dim=32, depth=2, pooling=pooling, positional_encoding=positions, patch_method=patch
    )
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(3, 3, 32, 32)), torch.tensor([0, 1, 2])
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_weight_decay_groups():
    model = VisionTransformer()
    optimizer = optimizer_for(model, load_config("configs/debug_cpu.yaml"))
    no_decay = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0
        for p in group["params"]
    }
    for name, p in model.named_parameters():
        excluded = p.ndim <= 1 or name == "cls_token" or name.startswith("pos.")
        assert (id(p) in no_decay) == excluded
