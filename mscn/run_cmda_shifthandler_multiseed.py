#!/usr/bin/env python3
"""Select target queries with ShiftHandler CBP, then train full CMDA."""
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


def parse_seeds(value):
    if "-" in value:
        start, end = (int(v) for v in value.split("-", 1))
        return range(start, end + 1)
    return [int(v) for v in value.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["job-light-ranges", "stats"], required=True)
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output_root", default="../results/cmda_shifthandler_multiseed/cardinality")
    p.add_argument("--split_root", default="../results/cardinality_multiseed")
    p.add_argument("--checkpoint_root", default="../results/sequential_drift/source_checkpoints")
    p.add_argument("--shift_handler_root", default=os.environ.get("CMDA_SHIFTHANDLER_ROOT", str(Path(__file__).resolve().parents[1] / "third_party/ShiftHandler")))
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--candidate_fraction", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=20260923)
    args = p.parse_args()

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
    output_root = Path(args.output_root).resolve()

    for seed in parse_seeds(args.seeds):
        run_dir = (output_root / args.dataset / f"budget_{args.budget}" /
                   "cmda_shifthandler" / f"seed_{seed:03d}")
        if (run_dir / "metrics.json").exists():
            print("SKIP", args.dataset, seed)
            continue
        selected, _ = select_shifthandler(
            stage, args.budget, seed, checkpoint, target_query,
            target_bitmap, selector_args)
        command = [
            sys.executable, "-u", "run.py", "--dataset", args.dataset,
            "--train_mode", "train", "--seed", str(seed),
            "--adaptation_budget", str(args.budget),
            "--selection_indices", ",".join(str(v) for v in selected),
            "--method_name", "cmda_shifthandler", "--contrastive", "--meta",
            "--epochs", str(args.epochs), "--batch_size", "512",
            "--alpha", "0.8", "--temperature", "2.0", "--lr", "0.001",
            "--device", args.device, "--output_root", str(output_root),
            "--target_candidate_fraction", str(args.candidate_fraction),
            "--split_seed", str(args.split_seed)]
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "shifthandler_selection.json").write_text(json.dumps(
            {"seed": seed, "selected_ids": selected, "budget": args.budget}, indent=2))


if __name__ == "__main__":
    main()
