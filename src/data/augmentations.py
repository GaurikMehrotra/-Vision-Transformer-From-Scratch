"""From-scratch batch Mixup and CutMix; torchvision spatial augmentations."""

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "synthetic": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def image_transform(dataset: str, size: int, augmentation: dict, train: bool) -> transforms.Compose:
    """PIL image -> normalized [3,H,W]; validation has no random transforms."""
    operations = []
    if train:
        if augmentation["random_crop"]:
            operations.append(transforms.RandomCrop(size, padding=augmentation["crop_padding"]))
        if augmentation["horizontal_flip"]:
            operations.append(transforms.RandomHorizontalFlip())
        if augmentation["randaugment"]:
            operations.append(
                transforms.RandAugment(
                    num_ops=augmentation["randaugment_ops"],
                    magnitude=augmentation["randaugment_magnitude"],
                )
            )
    operations += [transforms.ToTensor(), transforms.Normalize(*STATS[dataset])]
    return transforms.Compose(operations)


def mixup(images: Tensor, targets: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    """Convex combination of [B,C,H,W] images and soft labels [B,K]."""
    if alpha <= 0:
        return images, targets
    lam = float(np.random.beta(alpha, alpha))
    order = torch.randperm(images.shape[0], device=images.device)
    return images * lam + images[order] * (1 - lam), targets * lam + targets[order] * (1 - lam)


def cutmix(images: Tensor, targets: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    """Replace a rectangle and weight labels by its actual clipped pixel area."""
    if alpha <= 0:
        return images, targets
    lam = float(np.random.beta(alpha, alpha))
    order = torch.randperm(images.shape[0], device=images.device)
    h, w = images.shape[-2:]
    ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + (cut_h + 1) // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + (cut_w + 1) // 2, w)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[order, :, y1:y2, x1:x2]
    actual_lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    return mixed, targets * actual_lam + targets[order] * (1 - actual_lam)


def batch_augment(
    images: Tensor, labels: Tensor, num_classes: int, settings: dict
) -> tuple[Tensor, Tensor]:
    """Select at most one of Mixup/CutMix per batch; return [B,K] targets."""
    targets = torch.nn.functional.one_hot(labels, num_classes).to(images.dtype)
    mix, cut = settings["mixup_alpha"], settings["cutmix_alpha"]
    if (
        mix < 0
        or cut < 0
        or not 0 <= settings["mix_probability"] <= 1
        or not 0 <= settings["mixup_choice_probability"] <= 1
    ):
        raise ValueError("Invalid batch augmentation settings")
    if np.random.random() >= settings["mix_probability"]:
        return images, targets
    # Preserve the probability assigned to each method when one is disabled.
    # Disabled branches become identity, rather than doubling the other method.
    if np.random.random() < settings["mixup_choice_probability"]:
        return mixup(images, targets, mix)
    return cutmix(images, targets, cut)
    return images, targets
