import pytest
import torch

from src.models.vit import VisionTransformer


@pytest.mark.slow
def test_overfit_single_batch_to_near_zero_loss():
    model = VisionTransformer(
        image_size=16, patch_size=4, embed_dim=32, depth=1, num_heads=4, num_classes=4, mlp_ratio=2
    )
    images = torch.randn(4, 3, 16, 16)
    labels = torch.arange(4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(150):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        if float(loss.detach()) < 0.005:
            break
    with torch.no_grad():
        logits = model(images)
        final_loss = float(torch.nn.functional.cross_entropy(logits, labels))
    assert final_loss < 0.01, final_loss
    assert torch.equal(logits.argmax(1), labels)
