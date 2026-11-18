#!/usr/bin/env python3
"""Static 20-seed ShiftHandler comparison for Bao cost estimation."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cost_experiment
import dataset
import model
import train
from run_cost_sequential_drift import select_shifthandler_aligned


def parse_seeds(value):
    if "-" in value:
        left, right = (int(x) for x in value.split("-", 1))
        return range(left, right + 1)
    return [int(x) for x in value.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["job-light-ranges", "stats"], required=True)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--output_root", default="../../results/shifthandler_static_multiseed/cost")
    parser.add_argument("--checkpoint_root", default="../../results/cost_sequential_drift/source_checkpoints")
    parser.add_argument("--shift_handler_root", default=os.environ.get("CMDA_SHIFTHANDLER_ROOT", str(Path(__file__).resolve().parents[2] / "third_party/ShiftHandler")))
    args = parser.parse_args()
    config = dataset.load_multiseed_dataset(args.dataset)
    _, _, pool_ids, _ = train.load_aligned_experience(
        config["adaptation_plans"], config["adaptation_labels"])
    stage = {"stage": 0, "arrival_indices": pool_ids}
    checkpoint = Path(args.checkpoint_root).resolve() / args.dataset / "seed_000"
    reg = model.BaoRegression(have_cache_data=True, alpha=0.5, lr=0.001)
    reg.load(str(checkpoint / "model"), 1)
    root = Path(args.output_root).resolve()
    for seed in parse_seeds(args.seeds):
        metrics = root / args.dataset / f"budget_{args.budget}" / "shifthandler" / f"seed_{seed:03d}" / "metrics.json"
        if metrics.exists():
            print("SKIP", seed); continue
        selected, _ = select_shifthandler_aligned(
            stage, args.budget, seed, reg, config["adaptation_plans"],
            config["adaptation_labels"], args.shift_handler_root, 1.0)
        command = [sys.executable, "-u", "baoctl.py", "--train_mode", "train",
                   "--dataset", args.dataset, "--seed", str(seed),
                   "--method_name", "shifthandler", "--output_root", str(root),
                   "--adaptation_budget", str(args.budget), "--epochs", str(args.epochs),
                   "--batch_size", "256", "--alpha", "0.5", "--lr", "0.001",
                   "--repeat", "1", "--selection_indices",
                   ",".join(str(x) for x in selected)]
        subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == "__main__":
    main()
