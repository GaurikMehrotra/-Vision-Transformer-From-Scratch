"""Evaluate a trusted selected checkpoint; test access requires --final-test."""

import argparse
from pathlib import Path

import torch

from .data.datasets import loader, test_dataset, training_datasets
from .train import evaluate_loader
from .utils import build_model, config_hash, device_for, save_json, seed_everything


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "validation",
    final_test: bool = False,
    device_name: str = "auto",
) -> dict:
    """Use validation by default; never use test scores to choose hyperparameters."""
    if split not in {"validation", "test"} or (split == "test" and not final_test):
        raise ValueError("Final test evaluation requires --split test --final-test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(config["seed"], config["training"]["deterministic"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = device_for(device_name)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = test_dataset(config) if split == "test" else training_datasets(config)[1]
    metrics = evaluate_loader(model, loader(dataset, config, False), device)
    return {
        "split": split,
        "dataset": config["data"]["dataset"],
        "seed": config["seed"],
        "model": config["model"]["name"],
        "checkpoint": str(checkpoint_path),
        "selected_epoch": checkpoint["epoch"] + 1,
        "config_sha256": config_hash(config),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.split, args.final_test, args.device)
    save_json(result, Path(args.output))
    print(result)


if __name__ == "__main__":
    main()
