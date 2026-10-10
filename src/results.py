"""Aggregate real completed experiments, reject duplicates and incompatible budgets."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .utils import config_hash, load_config, save_json


def summarize(
    root: str | Path,
    output: str | Path,
    minimum_seeds: int = 3,
    include_synthetic: bool = False,
    metric: str = "validation",
) -> list[dict]:
    """Aggregate validation or final test; exclude subset/synthetic headline claims."""
    if metric not in {"validation", "test"}:
        raise ValueError("metric must be validation or test")
    if minimum_seeds < 1:
        raise ValueError("minimum_seeds must be positive")
    groups = defaultdict(list)
    configs = {}
    skipped = []
    for path in sorted(Path(root).rglob("summary.json")):
        summary = json.loads(path.read_text())
        if summary["status"] != "complete":
            skipped.append({"path": str(path), "reason": "interrupted"})
            continue
        if summary["dataset"] == "synthetic" and not include_synthetic:
            skipped.append({"path": str(path), "reason": "synthetic"})
            continue
        config = load_config(path.parent / "config.yaml")
        if config_hash(config) != summary["config_sha256"]:
            raise ValueError(f"Config fingerprint mismatch: {path}")
        if (
            summary["seed"] != config["seed"]
            or summary["dataset"] != config["data"]["dataset"]
            or summary["model"] != config["model"]["name"]
            or summary["epochs_completed"] != config["training"]["epochs"]
            or summary["epochs_planned"] != config["training"]["epochs"]
        ):
            raise ValueError(f"Inconsistent completed-run metadata: {path}")
        identity = json.loads(json.dumps(config))
        identity.pop("seed")
        identity.pop("output_dir", None)
        key = config_hash(identity)
        if metric == "test":
            final_path = path.parent / "test.json"
            if not final_path.exists():
                skipped.append({"path": str(path), "reason": "final test not evaluated"})
                continue
            final = json.loads(final_path.read_text())
            if final["split"] != "test" or final["config_sha256"] != summary["config_sha256"]:
                raise ValueError("Final test does not match saved configuration")
            if final["selected_epoch"] != summary["best_epoch"]:
                raise ValueError("Final evaluation did not use the selected best epoch")
            summary["test_accuracy"] = final["accuracy"]
        groups[key].append(summary)
        configs[key] = config
    rows = []
    for identity, entries in groups.items():
        seeds = [entry["seed"] for entry in entries]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed runs for the same configuration")
        if len({entry["split_sha256"] for entry in entries}) != 1:
            raise ValueError("Different data splits cannot be pooled")
        values = [
            entry["best_validation_accuracy" if metric == "validation" else "test_accuracy"]
            for entry in entries
        ]
        cfg = configs[identity]
        rows.append(
            {
                "model": cfg["model"]["name"],
                "dataset": cfg["data"]["dataset"],
                "config_id": identity,
                "seeds": ",".join(map(str, sorted(seeds))),
                "n": len(values),
                "mean_accuracy": statistics.mean(values),
                "metric": metric,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "headline_eligible": len(values) >= max(3, minimum_seeds)
                and cfg["data"]["dataset"] != "synthetic"
                and cfg["data"]["train_subset"] is None
                and cfg["data"]["validation_subset"] is None,
                "epochs": cfg["training"]["epochs"],
                "model_config": json.dumps(cfg["model"], sort_keys=True),
            }
        )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    save_json({"rows": rows, "skipped": skipped, "metric": metric}, destination / "results.json")
    fields = (
        list(rows[0])
        if rows
        else ["model", "dataset", "n", "mean_accuracy", "sample_std", "headline_eligible"]
    )
    with (destination / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = [
        f"Metric: {metric} accuracy. Standard deviation uses n-1.\n",
        "| Model / config | Dataset | Seeds | Accuracy mean ± sample std | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        deviation = (
            f"{100 * row['sample_std']:.2f}" if row["sample_std"] is not None else "undefined (n=1)"
        )
        table.append(
            f"| {row['model']} / {row['config_id'][:8]} | {row['dataset']} | {row['n']} | "
            f"{100 * row['mean_accuracy']:.2f} ± {deviation}% | "
            f"{'3+ seeds' if row['headline_eligible'] else 'diagnostic only'} |"
        )
    if not rows:
        table.append(
            "| No completed real experiments found | — | 0 | Not measured | Pending GPU runs |"
        )
    (destination / "RESULTS.md").write_text("\n".join(table) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/experiments")
    parser.add_argument("--output", default="reports/generated/results")
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--metric", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--include-synthetic", action="store_true", help="Diagnostic only, never headline results"
    )
    args = parser.parse_args()
    rows = summarize(
        args.root, args.output, args.minimum_seeds, args.include_synthetic, args.metric
    )
    print(f"{len(rows)} configurations summarized")


if __name__ == "__main__":
    main()
