"""Attention, rollout, position similarity, and patch filters from a ViT checkpoint."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data.augmentations import STATS
from .data.datasets import training_datasets
from .utils import build_model, device_for, save_json, seed_everything


def attention_rollout(maps: list[torch.Tensor]) -> torch.Tensor:
    """[layers][B,H,N,N] -> [B,N,N], average heads, add/normalize identity.

    This is a heuristic composition of attention routing, not a causal
    explanation: it ignores value/output projections, MLPs and gradient signs.
    """
    if not maps:
        raise ValueError("At least one attention map required")
    n = maps[0].shape[-1]
    identity = torch.eye(n, device=maps[0].device)[None]
    result = identity.expand(maps[0].shape[0], -1, -1)
    for weights in maps:
        augmented = weights.mean(dim=1) + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        result = augmented @ result
    return result


def save_heatmap(values: np.ndarray, path: Path, title: str) -> None:
    """Save a labeled standalone figure; never manufacture a web preview."""
    fig, axis = plt.subplots(figsize=(6, 5))
    handle = axis.imshow(values, cmap="viridis")
    axis.set_title(title)
    fig.colorbar(handle, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def visualize_checkpoint(
    checkpoint_path: str | Path, output: str | Path, index: int = 0, device_name: str = "auto"
) -> dict:
    """Use one validation image, never the test set; save raw maps and metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["model"]["name"] != "vit":
        raise ValueError("Attention visualizations require a ViT checkpoint")
    seed_everything(config["seed"], config["training"]["deterministic"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = device_for(device_name)
    model = build_model(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    _, validation, _ = training_datasets(config)
    image, label = validation[index]
    logits, maps = model(image[None].to(device), return_attention=True)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    m = config["model"]
    grid = m["image_size"] // m["patch_size"]
    prefix = int(m["pooling"] == "cls")
    mean, std = STATS[config["data"]["dataset"]]
    display = (
        (image * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None])
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    plt.imsave(destination / "image.png", display)
    raw = {}
    for layer, attention in enumerate(maps):
        values = attention[0].cpu().numpy()
        raw[f"layer_{layer}"] = values
        heads = values.shape[0]
        columns = min(4, heads)
        rows = (heads + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
        for head, axis in enumerate(axes.flat):
            if head >= heads:
                axis.axis("off")
                continue
            # GAP has no privileged query; average all patch-query distributions.
            patch_attention = values[head, 0, prefix:] if prefix else values[head].mean(axis=0)
            axis.imshow(patch_attention.reshape(grid, grid), cmap="viridis")
            axis.set_title(f"Layer {layer + 1}, head {head + 1}")
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(destination / f"attention_layer_{layer + 1:02d}.png", dpi=150)
        plt.close(fig)
    np.savez_compressed(destination / "attention.npz", **raw)
    rollout = attention_rollout(maps)[0].cpu().numpy()
    patch_scores = rollout[0, prefix:] if prefix else rollout.mean(axis=0)
    heatmap = patch_scores.reshape(grid, grid)
    save_heatmap(heatmap, destination / "rollout.png", "Attention rollout (routing heuristic)")
    fig, axis = plt.subplots()
    axis.imshow(display)
    expanded = np.repeat(np.repeat(heatmap, m["patch_size"], axis=0), m["patch_size"], axis=1)
    axis.imshow(expanded, cmap="inferno", alpha=0.45)
    axis.axis("off")
    fig.savefig(destination / "rollout_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if m["positional_encoding"] == "learned":
        positions = torch.nn.functional.normalize(model.pos.encoding[0, prefix:].float(), dim=-1)
        similarities = (positions @ positions.T).cpu().numpy()
        save_heatmap(
            similarities,
            destination / "position_similarity.png",
            "Learned patch-position cosine similarity",
        )
        center = (grid // 2) * grid + grid // 2
        save_heatmap(
            similarities[center].reshape(grid, grid),
            destination / "position_center.png",
            "Similarity to central patch position",
        )
    weights = model.patch_embed.proj.weight.detach().cpu()
    if m["patch_method"] == "unfold":
        weights = weights.reshape(
            m["embed_dim"], m["in_channels"], m["patch_size"], m["patch_size"]
        )
    filters = weights[: min(16, len(weights))].permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, axis in enumerate(axes.flat):
        axis.axis("off")
        if idx < len(filters):
            values = filters[idx]
            axis.imshow((values - values.min()) / max(float(values.max() - values.min()), 1e-8))
            axis.set_title(f"Filter {idx}")
    fig.tight_layout()
    fig.savefig(destination / "patch_filters.png", dpi=150)
    plt.close(fig)
    metadata = {
        "dataset": config["data"]["dataset"],
        "validation_index": index,
        "label": int(label),
        "prediction": int(logits.argmax(1)),
        "checkpoint": str(checkpoint_path),
        "selected_epoch": checkpoint["epoch"] + 1,
        "pooling": m["pooling"],
        "positional_encoding": m["positional_encoding"],
    }
    save_json(metadata, destination / "metadata.json")
    (destination / "ANALYSIS.md").write_text(
        "# Figure interpretation\n\n"
        f"Dataset: {metadata['dataset']}; true class {label}, predicted class {metadata['prediction']}.\n\n"
        "Head maps show where the CLS query routes attention (or mean patch-query attention for GAP). "
        "Brightness is attention probability, not evidence that a region caused the prediction. "
        "Rollout averages heads and composes identity-adjusted attention; it omits MLPs and value projections. "
        "Position cosine similarity can reveal spatial organization but is not proof of useful localization. "
        "Patch filters show signed RGB projection coefficients rescaled independently for display. "
        "They are not convolutional feature activations. Compare multiple correctly and incorrectly "
        "classified validation examples before drawing conclusions.\n\n"
        "These files were generated from the named checkpoint. A synthetic/debug run cannot support "
        "claims about learned semantics, benchmark accuracy, or generalization.\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="reports/generated/figures")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(visualize_checkpoint(args.checkpoint, args.output, args.index, args.device))


if __name__ == "__main__":
    main()
