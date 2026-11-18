"""Select Static-Upweight by inner CV on adaptation queries, then retrain."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def parse_list(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def run(command, cwd, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as stream:
        completed = subprocess.run(command, cwd=str(cwd), stdout=stream,
                                   stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError("Failed command; see {}".format(log_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cardinality", "cost"], required=True)
    parser.add_argument("--datasets", default="job-light-ranges,stats")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--weights", default="1,10,50,100,500")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--adaptation_budget", type=int, default=10)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    datasets = parse_list(args.datasets, str)
    seeds = parse_list(args.seeds, int)
    weights = parse_list(args.weights, float)
    if args.folds < 2 or args.folds > args.adaptation_budget:
        raise ValueError("folds must be between 2 and adaptation_budget")

    if args.task == "cardinality":
        cwd = repo / "mscn"
        epochs = args.epochs or 50
        batch_size = args.batch_size or 512
        base = [sys.executable, "-u", "run.py"]
        common = ["--target_candidate_fraction", "0.2", "--split_seed", "20260923",
                  "--device", "cpu", "--num_samples", "1000", "--alpha", "0.8",
                  "--lr", "0.001", "--repeat", "1"]
    else:
        cwd = repo / "BaoForPostgreSQL" / "bao_server"
        epochs = args.epochs or 100
        batch_size = args.batch_size or 256
        base = [sys.executable, "-u", "baoctl.py"]
        common = ["--alpha", "0.5", "--lr", "0.001", "--repeat", "1"]

    for dataset in datasets:
        for seed in seeds:
            selection_file = (output_root / dataset /
                              "budget_{}".format(args.adaptation_budget) /
                              "selections" / "seed_{:03d}.json".format(seed))
            if not selection_file.exists():
                prepare = base + ["--dataset", dataset, "--train_mode", "train",
                    "--seed", str(seed), "--adaptation_budget", str(args.adaptation_budget),
                    "--output_root", str(output_root), "--method_name", "static_prepare",
                    "--prepare_selection_only"] + common
                run(prepare, cwd, output_root / "static_upweight_logs" /
                    "{}_seed_{:03d}_prepare.log".format(dataset, seed))
            selection = json.loads(selection_file.read_text())
            ids = np.asarray(selection["adaptation_indices"], dtype=int)
            folds = [part.tolist() for part in np.array_split(ids, args.folds)]
            scores = {}
            for weight in weights:
                fold_scores = []
                for fold_index, validation_ids in enumerate(folds):
                    validation_set = set(validation_ids)
                    train_ids = [int(item) for item in ids if item not in validation_set]
                    method = "static_cv_w{}_fold{}".format(
                        str(weight).replace(".", "p"), fold_index)
                    command = base + ["--dataset", dataset, "--train_mode", "train",
                        "--seed", str(seed), "--adaptation_budget", str(args.adaptation_budget),
                        "--output_root", str(output_root), "--method_name", method,
                        "--epochs", str(epochs), "--batch_size", str(batch_size),
                        "--static_target_weight", str(weight),
                        "--static_train_indices", ",".join(map(str, train_ids)),
                        "--static_validation_indices", ",".join(map(str, validation_ids))] + common
                    log = output_root / "static_upweight_logs" / (
                        "{}_seed_{:03d}_w{}_fold{}.log".format(
                            dataset, seed, weight, fold_index))
                    run(command, cwd, log)
                    metrics = json.loads((output_root / dataset /
                        "budget_{}".format(args.adaptation_budget) / method /
                        "seed_{:03d}".format(seed) / "metrics.json").read_text())
                    fold_scores.append(metrics["mean_qerror"])
                scores[str(weight)] = fold_scores
            selected = min(weights, key=lambda weight: (np.mean(scores[str(weight)]), weight))
            final = base + ["--dataset", dataset, "--train_mode", "train",
                "--seed", str(seed), "--adaptation_budget", str(args.adaptation_budget),
                "--output_root", str(output_root), "--method_name", "static_upweight",
                "--epochs", str(epochs), "--batch_size", str(batch_size),
                "--static_target_weight", str(selected)] + common
            run(final, cwd, output_root / "static_upweight_logs" /
                "{}_seed_{:03d}_final.log".format(dataset, seed))
            result_dir = (output_root / dataset /
                          "budget_{}".format(args.adaptation_budget) /
                          "static_upweight" / "seed_{:03d}".format(seed))
            (result_dir / "weight_selection.json").write_text(json.dumps({
                "selection_metric": "mean_qerror",
                "folds": folds,
                "fold_scores": scores,
                "mean_scores": {key: float(np.mean(value)) for key, value in scores.items()},
                "selected_weight": selected,
            }, indent=2, sort_keys=True))
            print(dataset, seed, "selected", selected)


if __name__ == "__main__":
    main()
