import pytest
import torch

from src.models.vit import VisionTransformer
from src.profile import profile_model, timm_reference
from src.utils import load_config


@pytest.mark.parametrize("pooling", ["cls", "avg"])
@pytest.mark.parametrize("positions", ["learned", "none"])
def test_timm_logits_and_profile(pooling, positions):
    config = load_config("configs/debug_cpu.yaml")
    config["model"].update(pooling=pooling, positional_encoding=positions)
    ours = VisionTransformer(**{k: v for k, v in config["model"].items() if k != "name"}).eval()
    reference = timm_reference(config).eval()
    state = reference.state_dict()
    copied = {}
    for key, value in ours.state_dict().items():
        if key == "pos.encoding":
            copied[key] = state["pos_embed"] if positions == "learned" else torch.zeros_like(value)
        else:
            copied[key] = state[key]
    ours.load_state_dict(copied, strict=True)
    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(ours(images), reference(images), atol=1e-5, rtol=1e-4)
    actual = profile_model(ours, images[:1])
    expected = profile_model(reference, images[:1])
    assert actual["parameters"] == expected["parameters"]
    assert actual["counted_flops"] == expected["counted_flops"]
