import torch

from src.models.attention import Attention


def test_matches_pytorch_attention_outputs_weights_and_gradients():
    ours = Attention(32, 4).double().eval()
    reference = torch.nn.MultiheadAttention(32, 4, batch_first=True).double().eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(ours.qkv.weight)
        reference.in_proj_bias.copy_(ours.qkv.bias)
        reference.out_proj.weight.copy_(ours.proj.weight)
        reference.out_proj.bias.copy_(ours.proj.bias)
    x = torch.randn(2, 9, 32, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, attention = ours(x, True)
    expected, weights = reference(y, y, y, need_weights=True, average_attn_weights=False)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(attention, weights, atol=1e-10, rtol=1e-9)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-10, rtol=1e-9)
