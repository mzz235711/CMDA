"""Train historical-source-only checkpoints for sequential drift experiments."""

import argparse
import json
from pathlib import Path
import shutil
import time

import run as mscn_run


METHOD_SUFFIXES = ("", "_contrastive", "_meta", "_contrastive_meta")


def train_dataset(args, dataset):
    if dataset == "job-light-ranges":
        paths = mscn_run.load_job_dataset()
    elif dataset == "stats":
        paths = mscn_run.load_stats_dataset()
    else:
        raise ValueError(dataset)
    (source_query, min_max, _target_query, _result, _save,
     _adaptation, _validation, source_bitmap, _target_bitmap,
     _adaptation_bitmap, _tables) = paths
    output = (Path(args.output_root).resolve() / dataset /
              "seed_{:03d}".format(args.seed))
    output.mkdir(parents=True, exist_ok=True)
    mscn_run.set_global_seed(args.seed)
    start = time.perf_counter()
    # The held-out row is drawn from the historical workload itself. repeat=0
    # means the optimization set contains each source query exactly once; no
    # target-workload query is used to construct this checkpoint.
    mscn_run.train_triplet(
        dataset, source_query, min_max, str(output), source_bitmap,
        source_query, source_bitmap, 0, 0,
        num_samples=args.num_samples, cuda=args.device,
        batch_size=args.batch_size, num_epochs=args.epochs,
        alpha=args.alpha, lr=args.lr, repeat=0,
        contrastive=False, meta=False, adaptation_indices=[0])
    elapsed = time.perf_counter() - start
    base_model = output / "{}_model_{}.pkl".format(dataset, args.alpha)
    base_structure = output / "{}_structure.pkl".format(dataset)
    if not base_model.exists() or not base_structure.exists():
        raise RuntimeError("Source checkpoint artifacts were not created")
    for suffix in METHOD_SUFFIXES[1:]:
        shutil.copy2(base_model, output / "{}_model_{}{}.pkl".format(
            dataset, args.alpha, suffix))
        shutil.copy2(base_structure, output / "{}_structure{}.pkl".format(
            dataset, suffix))
    manifest = {
        "dataset": dataset,
        "seed": args.seed,
        "source_query_file": str(Path(source_query).resolve()),
        "source_bitmap_file": str(Path(source_bitmap).resolve()),
        "target_queries_used": 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "alpha_for_filename": args.alpha,
        "training_seconds": elapsed,
        "checkpoint_folder": str(output),
        "aliases_share_identical_initial_weights": True,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    print("Completed source-only checkpoint:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="job-light-ranges,stats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--output_root", default="../results/sequential_drift/source_checkpoints")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    for dataset in (item.strip() for item in args.datasets.split(",")):
        train_dataset(args, dataset)


if __name__ == "__main__":
    main()
