"""Disjoint stratified validation split from training data; test loaded separately."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from ..utils import Config, seed_worker
from .augmentations import image_transform


class SyntheticDataset(Dataset):
    """Deterministic fake RGB images for infrastructure smoke tests, never results."""

    def __init__(self, count: int, size: int, classes: int, seed: int, transform=None) -> None:
        generator = np.random.default_rng(seed)
        self.images = generator.integers(0, 256, (count, size, size, 3), dtype=np.uint8)
        self.targets = (np.arange(count) % classes).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        image = Image.fromarray(self.images[index])
        return self.transform(image), self.targets[index]


def raw_dataset(config: Config, train: bool, transform) -> Dataset:
    """CIFAR via torchvision or clearly marked synthetic data for smoke checks."""
    data = config["data"]
    if data["dataset"] == "synthetic":
        return SyntheticDataset(
            data["synthetic_train_size"] if train else data["synthetic_test_size"],
            data["image_size"],
            config["model"]["num_classes"],
            data["split_seed"] + int(not train),
            transform,
        )
    cls = datasets.CIFAR10 if data["dataset"] == "cifar10" else datasets.CIFAR100
    return cls(
        root=str(Path(data["root"]).expanduser()),
        train=train,
        download=data["download"],
        transform=transform,
    )


def stratified_split(
    targets: list[int], validation_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Proportional class allocation with exact total and a fixed independent seed."""
    labels = np.asarray(targets)
    if not 0 < validation_size < len(labels):
        raise ValueError("Validation size must be positive and smaller than training set")
    generator = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    ideal = counts * validation_size / len(labels)
    allocations = np.floor(ideal).astype(int)
    for index in np.argsort(-(ideal - allocations))[: validation_size - int(allocations.sum())]:
        allocations[index] += 1
    training, validation = [], []
    for cls, count in zip(classes, allocations, strict=True):
        indices = generator.permutation(np.flatnonzero(labels == cls)).tolist()
        validation.extend(indices[:count])
        training.extend(indices[count:])
    return generator.permutation(training).tolist(), generator.permutation(validation).tolist()


def training_datasets(config: Config) -> tuple[Dataset, Dataset, dict]:
    """Build train/validation only; never instantiate the test set here."""
    data = config["data"]
    training = raw_dataset(
        config,
        True,
        image_transform(data["dataset"], data["image_size"], config["augmentation"], True),
    )
    validation = raw_dataset(
        config,
        True,
        image_transform(data["dataset"], data["image_size"], config["augmentation"], False),
    )
    train_indices, val_indices = stratified_split(
        training.targets, data["validation_size"], data["split_seed"]
    )
    if data["train_subset"] is not None:
        if not 0 < data["train_subset"] <= len(train_indices):
            raise ValueError("Invalid train_subset")
        train_indices = train_indices[: data["train_subset"]]
    if data["validation_subset"] is not None:
        if not 0 < data["validation_subset"] <= len(val_indices):
            raise ValueError("Invalid validation_subset")
        val_indices = val_indices[: data["validation_subset"]]
    identity = hashlib.sha256(
        np.asarray(train_indices + [-1] + val_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    return (
        Subset(training, train_indices),
        Subset(validation, val_indices),
        {"train_indices": train_indices, "validation_indices": val_indices, "sha256": identity},
    )


def test_dataset(config: Config) -> Dataset:
    """Explicit final-report entry point; called only by final test evaluation."""
    data = config["data"]
    return raw_dataset(
        config,
        False,
        image_transform(data["dataset"], data["image_size"], config["augmentation"], False),
    )


def loader(dataset: Dataset, config: Config, train: bool, epoch: int = 0) -> DataLoader:
    """Recreated each epoch with an epoch-specific seed for exact boundary resume."""
    settings = config["training"]
    generator = torch.Generator().manual_seed(config["seed"] + epoch + (0 if train else 100000))
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=train,
        num_workers=settings["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )
