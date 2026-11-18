#!/usr/bin/env python3
"""Run the official ShiftHandler selector on the fixed static target pool."""
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import run as mscn_run
from dataset import load_job_dataset, load_stats_dataset
from run_sequential_drift import select_shifthandler


def paths(dataset):
    values = load_job_dataset() if dataset == "job-light-ranges" else load_stats_dataset()
    return values[2], values[8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["job-light-ranges", "stats"], required=True)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output_root", default="../results/shifthandler_static_multiseed/cardinality")
    parser.add_argument("--split_root", default="../results/cardinality_multiseed")
    parser.add_argument("--checkpoint_root", default="../results/sequential_drift/source_checkpoints")
    parser.add_argument("--shift_handler_root", default=os.environ.get("CMDA_SHIFTHANDLER_ROOT", str(Path(__file__).resolve().parents[1] / "third_party/ShiftHandler")))
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--candidate_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=20260923)
    args = parser.parse_args()
    if "-" in args.seeds:
        start, end = (int(v) for v in args.seeds.split("-", 1))
        seeds = range(start, end + 1)
    else:
        seeds = [int(v) for v in args.seeds.split(",")]

    target_query, target_bitmap = paths(args.dataset)
    split, _ = mscn_run.load_or_create_target_split(
        args.split_root, args.dataset, target_query,
        args.candidate_fraction, args.split_seed)
    checkpoint = Path(args.checkpoint_root) / args.dataset / "seed_000"
    selector_args = SimpleNamespace(
        shift_handler_root=args.shift_handler_root, dataset=args.dataset,
        device=args.device, batch_size=512, num_samples=1000,
        alpha=0.8, shift_concentration=1.0)
    stage = {"stage": 0, "arrival_indices": split["candidate_indices"]}
    for seed in seeds:
        selected, _ = select_shifthandler(
            stage, args.budget, seed, checkpoint, target_query,
            target_bitmap, selector_args)
        command = [
            sys.executable, "run.py", "--dataset", args.dataset,
            "--train_mode", "train",
            "--seed", str(seed), "--adaptation_budget", str(args.budget),
            "--selection_indices", ",".join(str(v) for v in selected),
            "--method_name", "shifthandler", "--epochs", str(args.epochs),
            "--batch_size", "512", "--alpha", "0.8", "--lr", "0.001",
            "--device", args.device, "--output_root", args.output_root,
            "--target_candidate_fraction", str(args.candidate_fraction),
            "--split_seed", str(args.split_seed)]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
        manifest = Path(args.output_root) / args.dataset / "budget_10" / "shifthandler" / f"seed_{seed:03d}" / "shifthandler_selection.json"
        manifest.write_text(json.dumps({"seed": seed, "selected_ids": selected}, indent=2))


if __name__ == "__main__":
    main()
