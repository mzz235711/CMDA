"""Train source-only Bao checkpoints for sequential-drift experiments."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cost_experiment
import dataset
import model
import train


def run(args, name):
    config = dataset.load_multiseed_dataset(name)
    experiences, labels, ids, alignment = train.load_aligned_experience(
        config["train_plans"], config["train_labels"])
    plans = [value[0] for value in experiences]
    output = (Path(args.output_root).resolve() / name /
              "seed_{:03d}".format(args.seed))
    output.mkdir(parents=True, exist_ok=True)
    cost_experiment.set_global_seed(args.seed)
    reg = model.BaoRegression(
        have_cache_data=True, verbose=True, alpha=args.alpha, lr=args.lr)
    started = time.perf_counter()
    # The final duplicate is used only as Bao's validation batch. Every item in
    # the optimization prefix comes from the historical source workload.
    reg.fit(plans + [plans[0]], labels + [labels[0]], train_num=len(plans),
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    reg.save(str(output / "model"), args.repeat)
    seconds = time.perf_counter() - started
    base = output / "model_{}_{}_{}".format(
        args.alpha, args.lr, args.repeat)
    cmda = output / "model_contrastive_meta_{}_{}_{}".format(
        args.alpha, args.lr, args.repeat)
    if cmda.exists():
        shutil.rmtree(str(cmda))
    shutil.copytree(str(base), str(cmda))
    (output / "source_manifest.json").write_text(json.dumps({
        "dataset": name, "seed": args.seed, "source_count": len(ids),
        "source_alignment": alignment, "target_queries_used": 0,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "alpha": args.alpha, "learning_rate": args.lr,
        "repeat": args.repeat, "training_seconds": seconds,
        "adapt_cmda_shifthandler_share_initial_weights": True,
    }, indent=2, sort_keys=True))
    print("Completed source-only Bao checkpoint:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="job-light-ranges,stats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--output_root", default="../../results/cost_sequential_drift/source_checkpoints")
    args = parser.parse_args()
    for name in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        run(args, name)


if __name__ == "__main__":
    main()
