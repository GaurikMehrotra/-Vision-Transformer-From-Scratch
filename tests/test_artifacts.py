import json
from pathlib import Path

import pytest
import torch

from src.results import summarize
from src.train import run_training
from src.utils import load_config
from src.visualize import attention_rollout, visualize_checkpoint


def test_attention_rollout_is_row_stochastic_and_layer_ordered():
    first = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]]]])
    second = torch.tensor([[[[0.6, 0.4], [0.3, 0.7]]]])
    identity = torch.eye(2)[None]
    expected = ((second.mean(1) + identity) / 2) @ ((first.mean(1) + identity) / 2)
    actual = attention_rollout([first, second])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, 2))
    with pytest.raises(ValueError):
        attention_rollout([])


@pytest.mark.slow
def test_checkpoint_figures_and_honest_aggregation(tmp_path):
    config = load_config("configs/smoke_synthetic.yaml")
    output = run_training(config, tmp_path / "run")
    metadata = visualize_checkpoint(output / "best.pt", tmp_path / "figures", device_name="cpu")
    assert metadata["dataset"] == "synthetic"
    for name in [
        "attention_layer_01.png",
        "attention_layer_02.png",
        "rollout_overlay.png",
        "patch_filters.png",
        "position_similarity.png",
        "metadata.json",
        "ANALYSIS.md",
    ]:
        assert (tmp_path / "figures" / name).stat().st_size > 0
    assert summarize(tmp_path / "run", tmp_path / "results") == []
    rows = summarize(tmp_path / "run", tmp_path / "diagnostic", include_synthetic=True)
    assert rows[0]["n"] == 1 and not rows[0]["headline_eligible"]
    # Duplicate seeds must never inflate confidence.
    import shutil

    shutil.copytree(output, tmp_path / "duplicate")
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize(tmp_path, tmp_path / "bad", include_synthetic=True)


def test_all_configuration_files_resolve():
    for path in Path("configs").rglob("*.yaml"):
        assert load_config(path)["training"]["epochs"] > 0


def test_walkthrough_notebook_executes():
    notebook = json.loads(Path("notebooks/01_components_walkthrough.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    assert namespace["logits"].shape == (2, 10)


def test_three_seed_mean_std_and_final_test_provenance(tmp_path):
    """Fabricated metadata is a temporary test fixture, never benchmark output."""
    import yaml

    from src.utils import config_hash, save_json

    for index, seed in enumerate([42, 43, 44]):
        config = load_config("configs/vit_tiny_cifar10.yaml")
        config["seed"] = seed
        folder = tmp_path / "fixture_runs" / f"seed{seed}"
        folder.mkdir(parents=True)
        (folder / "config.yaml").write_text(yaml.safe_dump(config))
        save_json(
            {
                "status": "complete",
                "dataset": "cifar10",
                "model": "vit",
                "seed": seed,
                "best_validation_accuracy": 0.5 + 0.1 * index,
                "best_epoch": 3,
                "epochs_completed": 100,
                "epochs_planned": 100,
                "config_sha256": config_hash(config),
                "split_sha256": "fixture-split",
            },
            folder / "summary.json",
        )
        save_json(
            {
                "split": "test",
                "accuracy": 0.4 + 0.1 * index,
                "config_sha256": config_hash(config),
                "selected_epoch": 3,
            },
            folder / "test.json",
        )
    validation = summarize(tmp_path / "fixture_runs", tmp_path / "validation")
    final = summarize(tmp_path / "fixture_runs", tmp_path / "final", metric="test")
    assert validation[0]["mean_accuracy"] == pytest.approx(0.6)
    assert validation[0]["sample_std"] == pytest.approx(0.1)
    assert validation[0]["headline_eligible"]
    assert final[0]["mean_accuracy"] == pytest.approx(0.5)
    altered = tmp_path / "fixture_runs/seed42/test.json"
    value = json.loads(altered.read_text())
    value["selected_epoch"] = 2
    altered.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="selected best epoch"):
        summarize(tmp_path / "fixture_runs", tmp_path / "bad", metric="test")


def test_from_scratch_model_dependencies_remain_basic_pytorch():
    import ast

    for file in Path("src/models").glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("timm") for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("timm")
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"MultiheadAttention", "scaled_dot_product_attention"}
