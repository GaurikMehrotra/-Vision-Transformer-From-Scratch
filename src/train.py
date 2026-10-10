"""Single-device training with epoch-boundary resume and validation-only selection."""

import argparse
import csv
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.augmentations import batch_augment
from .data.datasets import loader, training_datasets
from .utils import (
    Config,
    atomic_save,
    build_model,
    config_hash,
    device_for,
    load_config,
    optimizer_for,
    restore_rng,
    rng_state,
    save_json,
    seed_everything,
)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, batches: DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate unaugmented labels; loss is unsmoothed cross entropy."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss)
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += len(labels)
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": loss_sum / total, "accuracy": correct / total, "count": total}


def learning_rate(
    step: int, total_steps: int, warmup_steps: int, peak: float, minimum: float
) -> float:
    """Linear warmup then cosine; final planned step reaches the minimum."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def run_training(
    config: Config,
    output: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after_epoch: int | None = None,
) -> Path:
    """Train using config; stop_after_epoch emulates interruption without changing budget.

    Epoch checkpoints include model/optimizer/scaler/RNG and split fingerprint.
    Exact resume is supported at epoch boundaries on the same stack/device.
    Checkpoints use pickle; load ONLY trusted checkpoints.
    """
    settings = config["training"]
    seed_everything(config["seed"], settings["deterministic"])
    torch.set_num_threads(settings["cpu_threads"])
    device = device_for(settings["device"])
    directory = Path(output or config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if resume is None and (directory / "last.pt").exists():
        raise FileExistsError("Run exists: use --resume or a new --output directory")
    model = build_model(config).to(device)
    optimizer = optimizer_for(model, config)
    use_amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    training, validation, split = training_datasets(config)
    expected_classes = {"cifar10": 10, "cifar100": 100}.get(config["data"]["dataset"])
    if expected_classes is not None and config["model"]["num_classes"] != expected_classes:
        raise ValueError("Model class count differs from dataset")
    start, best, best_epoch, global_step, history = 0, -1.0, -1, 0, []
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if config_hash(checkpoint["config"]) != config_hash(config):
            raise ValueError("Resume config differs: preserve original schedule/model/data/seed")
        if checkpoint["split"]["sha256"] != split["sha256"]:
            raise ValueError("Dataset split changed")
        if checkpoint["device_type"] != device.type:
            raise ValueError("Exact resume requires the same device type")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start = checkpoint["epoch"] + 1
        best, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        global_step, history = checkpoint["global_step"], checkpoint["history"]
        restore_rng(checkpoint["rng"])
        # Keep a portable copy of best-so-far weights even when resuming elsewhere.
        atomic_save(
            {
                "model": checkpoint["best_model"],
                "config": config,
                "epoch": best_epoch,
                "validation_accuracy": best,
            },
            directory / "best.pt",
        )
    best_weights = checkpoint["best_model"] if resume is not None else None
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(split, directory / "split.json")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "uncommitted"
    save_json(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "git_commit": revision,
            "config_sha256": config_hash(config),
            "synthetic": config["data"]["dataset"] == "synthetic",
        },
        directory / "environment.json",
    )
    steps_per_epoch = math.ceil(len(training) / settings["batch_size"])
    total_steps = steps_per_epoch * settings["epochs"]
    warmup_steps = steps_per_epoch * settings["warmup_epochs"]
    with SummaryWriter(
        str(directory / "tensorboard"), purge_step=start if resume else None
    ) as writer:
        for epoch in range(start, settings["epochs"]):
            began = time.perf_counter()
            model.train()
            loss_sum, count = 0.0, 0
            for images, labels in loader(training, config, True, epoch):
                images, labels = images.to(device), labels.to(device)
                images, targets = batch_augment(
                    images, labels, config["model"]["num_classes"], config["augmentation"]
                )
                lr = learning_rate(
                    global_step, total_steps, warmup_steps, settings["lr"], settings["min_lr"]
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss = torch.nn.functional.cross_entropy(
                        model(images), targets, label_smoothing=settings["label_smoothing"]
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {global_step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(labels)
                count += len(labels)
                global_step += 1
            metrics = evaluate_loader(model, loader(validation, config, False, epoch), device)
            row = {
                "epoch": epoch + 1,
                "train_loss": loss_sum / count,
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
                "lr": lr,
                "seconds": time.perf_counter() - began,
            }
            history.append(row)
            for key in ("train_loss", "validation_loss", "validation_accuracy", "lr"):
                writer.add_scalar(key, row[key], epoch)
            writer.flush()
            if metrics["accuracy"] > best:
                best, best_epoch = metrics["accuracy"], epoch
                best_weights = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                atomic_save(
                    {
                        "model": best_weights,
                        "config": config,
                        "epoch": epoch,
                        "validation_accuracy": best,
                    },
                    directory / "best.pt",
                )
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_accuracy": best,
                "best_epoch": best_epoch,
                "best_model": best_weights,
                "config": config,
                "split": split,
                "rng": rng_state(),
                "device_type": device.type,
                "history": history,
            }
            atomic_save(checkpoint, directory / "last.pt")
            # last.pt is atomically refreshed every epoch; avoid unbounded checkpoint storage.
            save_json(history, directory / "metrics.json")
            with (directory / "metrics.csv").open("w", newline="") as handle:
                writer_csv = csv.DictWriter(handle, fieldnames=list(row))
                writer_csv.writeheader()
                writer_csv.writerows(history)
            print(
                f"epoch {epoch + 1}/{settings['epochs']} loss={row['train_loss']:.4f} val_acc={metrics['accuracy']:.4f}",
                flush=True,
            )
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                break
    save_json(
        {
            "status": "complete" if len(history) == settings["epochs"] else "interrupted",
            "dataset": config["data"]["dataset"],
            "model": config["model"]["name"],
            "seed": config["seed"],
            "best_validation_accuracy": best,
            "best_epoch": best_epoch + 1,
            "epochs_completed": len(history),
            "epochs_planned": settings["epochs"],
            "config_sha256": config_hash(config),
            "split_sha256": split["sha256"],
        },
        directory / "summary.json",
    )
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--resume", help="Trusted last.pt checkpoint; same original config required"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["data"]["dataset"] != "synthetic":
        # Run the full correctness suite before real-data training.
        import sys

        subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )
    if args.seed is not None:
        config["seed"] = args.seed
    run_training(config, args.output, args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
