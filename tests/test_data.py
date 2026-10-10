import numpy as np
import torch

from src.data.augmentations import batch_augment, cutmix, mixup
from src.data.datasets import loader, stratified_split, training_datasets
from src.utils import load_config, seed_everything


def test_split_disjoint_and_stratified():
    labels = np.repeat(np.arange(10), 100).tolist()
    train, validation = stratified_split(labels, 200, 17)
    assert len(train) == 800 and len(validation) == 200
    assert not set(train) & set(validation)
    assert set(train + validation) == set(range(1000))
    assert np.bincount(np.asarray(labels)[validation]).tolist() == [20] * 10
    assert (train, validation) == stratified_split(labels, 200, 17)


def test_mixup_soft_labels_correspond_to_image_mixture():
    images = torch.eye(4).reshape(4, 1, 2, 2)
    targets = torch.eye(4)
    mixed, labels = mixup(images, targets, 0.4)
    torch.testing.assert_close(mixed.flatten(1), labels)
    torch.testing.assert_close(labels.sum(1), torch.ones(4))
    assert torch.all(labels >= 0)


def test_cutmix_effective_area_matches_labels():
    images = torch.arange(4, dtype=torch.float32)[:, None, None, None].expand(4, 1, 16, 16).clone()
    targets = torch.eye(4)
    for seed in range(10):
        seed_everything(seed)
        mixed, labels = cutmix(images, targets, 1.0)
        torch.testing.assert_close(
            mixed.mean((1, 2, 3)), labels @ torch.arange(4, dtype=torch.float32)
        )
        torch.testing.assert_close(labels.sum(1), torch.ones(4))
    torch.testing.assert_close(images[:, 0, 0, 0], torch.arange(4, dtype=torch.float32))


def test_augmentation_can_be_disabled():
    cfg = load_config("configs/smoke_synthetic.yaml")
    images, labels = torch.randn(4, 3, 32, 32), torch.arange(4)
    cfg["augmentation"]["mix_probability"] = 0
    actual, targets = batch_augment(images, labels, 10, cfg["augmentation"])
    assert actual is images
    assert torch.equal(targets.argmax(1), labels)


def test_workers_and_validation_transforms_reproducible():
    config = load_config("configs/smoke_synthetic.yaml")
    config["training"]["num_workers"] = 2
    training, validation, split = training_datasets(config)
    assert not set(split["train_indices"]) & set(split["validation_indices"])
    first = next(iter(loader(training, config, True, epoch=3)))
    second = next(iter(loader(training, config, True, epoch=3)))
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    torch.testing.assert_close(validation[0][0], validation[0][0], atol=0, rtol=0)
