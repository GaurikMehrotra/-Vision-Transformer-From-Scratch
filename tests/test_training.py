import pytest
import torch

from src.evaluate import evaluate_checkpoint
from src.train import learning_rate, run_training
from src.utils import load_config


@pytest.mark.slow
def test_resume_matches_uninterrupted_model_and_optimizer(tmp_path):
    config = load_config("configs/smoke_synthetic.yaml")
    full = run_training(config, tmp_path / "full")
    interrupted = run_training(config, tmp_path / "interrupted", stop_after_epoch=1)
    resumed = run_training(config, tmp_path / "resumed", resume=interrupted / "last.pt")
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    assert a["global_step"] == b["global_step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][key][name], atol=0, rtol=0)
    assert a["history"][-1]["validation_accuracy"] == b["history"][-1]["validation_accuracy"]
    with pytest.raises(ValueError, match="Final test"):
        evaluate_checkpoint(resumed / "best.pt", split="test")
    result = evaluate_checkpoint(resumed / "best.pt")
    assert result["split"] == "validation" and result["count"] == 16
    config["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="Resume config"):
        run_training(config, tmp_path / "bad", resume=interrupted / "last.pt")


def test_learning_rate_schedule():
    rates = [learning_rate(s, 10, 2, 1.0, 0.01) for s in range(10)]
    assert rates[0] == 0.5 and rates[1] == rates[2] == 1.0
    assert rates[-1] == pytest.approx(0.01)
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))


@pytest.mark.slow
def test_resnet_smoke_training_and_best_checkpoint(tmp_path):
    config = load_config("configs/smoke_synthetic.yaml")
    config["model"]["name"] = "resnet"
    config["model"]["width"] = 4
    config["model"]["blocks_per_stage"] = 1
    config["training"]["epochs"] = 1
    path = run_training(config, tmp_path / "resnet")
    result = evaluate_checkpoint(path / "best.pt", device_name="cpu")
    assert result["model"] == "resnet" and result["count"] == 16
    assert 0 <= result["accuracy"] <= 1
