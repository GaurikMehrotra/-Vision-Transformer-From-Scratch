"""Configuration, reproducibility, device selection, and atomic artifact helpers."""

import copy
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from .baselines.resnet_small import ResNetSmall
from .models.vit import VisionTransformer

Config = dict[str, Any]


def load_config(path: str | Path) -> Config:
    """Read YAML with optional relative `extends`, recursively merging mappings."""

    def read(current: Path, seen: set[Path]) -> Config:
        current = current.resolve()
        if current in seen:
            raise ValueError("Configuration inheritance cycle")
        with current.open() as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a mapping")
        parent = values.pop("extends", None)
        base = read(current.parent / parent, seen | {current}) if parent else {}
        return merge_config(base, values)

    config = read(Path(path), set())
    for section in ("data", "model", "training", "augmentation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    training = config["training"]
    if training["epochs"] <= 0 or training["batch_size"] <= 0 or training["num_workers"] < 0:
        raise ValueError("Invalid training lengths or worker count")
    if training["lr"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Learning rate must be positive; decay nonnegative")
    if not 0 <= training["warmup_epochs"] < training["epochs"]:
        raise ValueError("warmup_epochs must be in [0,epochs)")
    if not 0 <= training["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0,1)")
    if not 0 <= training["min_lr"] <= training["lr"]:
        raise ValueError("min_lr must be in [0,lr]")
    if config["data"]["dataset"] not in {"cifar10", "cifar100", "synthetic"}:
        raise ValueError("Supported datasets: cifar10, cifar100, synthetic (smoke only)")
    if config["data"]["dataset"] != "synthetic" and config["data"]["image_size"] != 32:
        raise ValueError("CIFAR images are 32x32; resizing is not implemented")
    if (
        config["model"]["name"] == "vit"
        and config["model"]["image_size"] != config["data"]["image_size"]
    ):
        raise ValueError("Model and dataset image sizes differ")
    return config


def merge_config(base: Config, updates: Config) -> Config:
    """Recursive merge without modifying either input."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = (
            merge_config(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch. Determinism may reduce GPU throughput."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Each DataLoader worker gets reproducible, distinct Python/NumPy seeds."""
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def device_for(name: str = "auto") -> torch.device:
    """auto selects available CUDA, otherwise CPU; explicit cuda must exist."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def build_model(config: Config) -> nn.Module:
    """Instantiate one configured model without third-party model internals."""
    values = dict(config["model"])
    name = values.pop("name")
    if name == "vit":
        return VisionTransformer(**values)
    if name == "resnet":
        # Shared ViT config fields are intentionally ignored for a baseline override.
        allowed = {"num_classes", "width", "blocks_per_stage"}
        return ResNetSmall(**{k: v for k, v in values.items() if k in allowed})
    raise ValueError(f"Unknown model: {name}")


def optimizer_for(model: nn.Module, config: Config) -> torch.optim.AdamW:
    """Exclude biases, norm scales, CLS and positional tokens from decay."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = parameter.ndim <= 1 or name == "cls_token" or name.startswith("pos.")
        (no_decay if excluded else decay).append(parameter)
    settings = config["training"]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings["lr"],
    )


def rng_state() -> dict[str, Any]:
    """Capture all RNG streams needed for single-process epoch-boundary resume."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore trusted checkpoint RNG state; CPU RNG byte tensors stay on CPU."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_save(value: Any, destination: Path) -> None:
    """Write in the destination directory then atomically replace the file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix="checkpoint-", suffix=".pt"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_json(value: Any, path: Path) -> None:
    """Serialize JSON, rejecting NaN metrics rather than silently publishing them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_hash(config: Config) -> str:
    """Stable config identity excluding output location; saved alongside metrics."""
    settings = copy.deepcopy(config)
    settings.pop("output_dir", None)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
