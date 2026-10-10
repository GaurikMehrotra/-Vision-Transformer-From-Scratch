"""Plan and run sequential multi-seed experiments with interruption recovery.

No test evaluation happens here. Hyperparameter/ablation comparisons use validation.
Use evaluate.py explicitly on the frozen headline models for final test reporting.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .utils import config_hash, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="runs/experiments")
    parser.add_argument(
        "--run", action="store_true", help="Actually train; default only saves a plan"
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds would falsely inflate experiment count")
    root = Path(args.output)
    plan = []
    seen = set()
    unique_configs = set()
    for path in args.configs:
        config = load_config(path)
        label = Path(path).stem
        if label in seen:
            parser.error("Configuration basenames must be unique")
        seen.add(label)
        identity = dict(config)
        identity.pop("seed")
        fingerprint = config_hash(identity)
        if fingerprint in unique_configs:
            print(
                f"Skipping equivalent configuration {label}; duplicate runs would inflate evidence."
            )
            continue
        unique_configs.add(fingerprint)
        for seed in args.seeds:
            destination = root / label / f"seed{seed}"
            plan.append(
                {
                    "config": path,
                    "seed": seed,
                    "run_dir": str(destination),
                    "dataset": config["data"]["dataset"],
                    "epochs": config["training"]["epochs"],
                }
            )
    save_json(plan, root / "plan.json")
    print(f"{len(plan)} runs planned. No wall-time estimate without a measured GPU epoch.")
    if not args.run:
        return
    # Correctness gates are required before real training, including slow checks.
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    for item in plan:
        destination = Path(item["run_dir"])
        summary_path = destination / "summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text())["status"] == "complete":
            expected = load_config(item["config"])
            expected["seed"] = item["seed"]
            if json.loads(summary_path.read_text())["config_sha256"] != config_hash(expected):
                raise ValueError(f"Existing completed run has different config: {destination}")
            print(f"Skipping completed {destination}")
            continue
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            item["config"],
            "--seed",
            str(item["seed"]),
            "--output",
            str(destination),
        ]
        if (destination / "last.pt").exists():
            command += ["--resume", str(destination / "last.pt")]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
