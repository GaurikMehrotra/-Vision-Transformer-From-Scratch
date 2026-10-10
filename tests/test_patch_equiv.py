import torch

from src.models.patch_embed import PatchEmbedding


def test_conv_matches_unfold_outputs_and_gradients():
    conv = PatchEmbedding(embed_dim=32).double()
    unfold = PatchEmbedding(embed_dim=32, method="unfold").double()
    with torch.no_grad():
        unfold.proj.weight.copy_(conv.proj.weight.flatten(1))
        unfold.proj.bias.copy_(conv.proj.bias)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = conv(x), unfold(y)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        conv.proj.weight.grad.flatten(1), unfold.proj.weight.grad, atol=1e-10, rtol=1e-10
    )
