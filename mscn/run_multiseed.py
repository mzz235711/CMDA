"""Run paired multi-seed MSCN cardinality experiments.

All methods for a dataset/seed read the same target-selection manifest created
by run.py. Runs are sequential by default to avoid GPU/CPU resource contention.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


METHOD_FLAGS = {
    "adapt": [],
    "cmda_c": ["--contrastive"],
    "cmda_m": ["--meta"],
    "cmda": ["--contrastive", "--meta"],
    "template_cl": ["--template_contrastive"],
    "random_cl": ["--contrastive", "--random_contrastive"],
    "robust": ["--robust"],
    "target": ["--target_only"],
    "all": ["--use_all_candidates"],
}


def parse_int_list(value):
    values = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.extend(range(start, end + 1))
        elif part:
            values.append(int(part))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="job-light-ranges,stats")
    parser.add_argument("--methods", default="adapt,cmda_c,cmda_m,cmda")
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--output_root", default="../results/cardinality_multiseed")
    parser.add_argument("--adaptation_budget", type=int, default=10)
    parser.add_argument("--target_candidate_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=20260923)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    seeds = parse_int_list(args.seeds)
    unknown = sorted(set(methods) - set(METHOD_FLAGS))
    if unknown:
        raise ValueError("Unknown methods: {}".format(unknown))

    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (script_dir / output_root).resolve()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
    env["MKL_NUM_THREADS"] = str(args.cpu_threads)

    plan = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                metrics_file = (output_root / dataset /
                                "budget_{}".format(args.adaptation_budget) /
                                method / "seed_{:03d}".format(seed) /
                                "metrics.json")
                if metrics_file.exists() and not args.force and not args.prepare_only:
                    status = "skip-complete"
                else:
                    status = "pending"
                plan.append({"dataset": dataset, "seed": seed,
                             "method": method, "status": status})
    output_root.mkdir(parents=True, exist_ok=True)
    plan_name = ("experiment_plan_template_cl.json"
                 if methods == ["template_cl"] else "experiment_plan.json")
    (output_root / plan_name).write_text(
        json.dumps(plan, indent=2, sort_keys=True))

    failures = []
    for item in plan:
        if item["status"] == "skip-complete":
            print("SKIP", item)
            continue
        command = [
            sys.executable, "-u", "run.py",
            "--dataset", item["dataset"],
            "--train_mode", "train",
            "--seed", str(item["seed"]),
            "--adaptation_budget", str(args.adaptation_budget),
            "--target_candidate_fraction", str(args.target_candidate_fraction),
            "--split_seed", str(args.split_seed),
            "--output_root", str(output_root),
            "--method_name", item["method"],
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--num_samples", str(args.num_samples),
            "--alpha", str(args.alpha),
            "--temperature", str(args.temperature),
            "--lr", str(args.lr),
            "--repeat", str(args.repeat),
        ] + METHOD_FLAGS[item["method"]]
        if args.prepare_only:
            command.append("--prepare_selection_only")
        log_dir = (output_root / item["dataset"] /
                   "budget_{}".format(args.adaptation_budget) / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "{}_seed_{:03d}.log".format(
            item["method"], item["seed"])
        print("RUN", " ".join(command))
        if args.dry_run:
            continue
        with log_file.open("w") as stream:
            completed = subprocess.run(
                command, cwd=str(script_dir), env=env,
                stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            failures.append({**item, "returncode": completed.returncode,
                             "log": str(log_file)})
            print("FAILED", failures[-1])
        else:
            print("DONE", item)

    if failures:
        (output_root / "failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True))
        raise SystemExit("{} runs failed".format(len(failures)))


if __name__ == "__main__":
    main()
