"""Sequential/repeated workload-drift evaluation for MSCN.

This driver is intentionally separate from run_multiseed.py and writes to an
isolated result root.  It constructs temporal stages from complete join-graph
families, adapts on either the first-arriving or randomly selected queries in
each stage, warm-starts every later stage from the preceding checkpoint, and
evaluates both recovery on the current stage and forgetting on a fixed source
replay set.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

import run as mscn_run


METHODS = {
    "adapt": (False, False),
    "cmda": (True, True),
    "shifthandler": (False, False),
}


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_workload_lines(path):
    with open(path) as stream:
        return [line.rstrip("\n") for line in stream]


def graph_signature(line):
    fields = line.split("#")
    if len(fields) < 3:
        raise ValueError("Malformed workload row: {}".format(line[:120]))
    tables = tuple(sorted(item.strip() for item in fields[0].split(",")
                          if item.strip()))
    joins = tuple(sorted(item.strip() for item in fields[1].split(",")
                         if item.strip()))
    return tables, joins


def stable_hash(values):
    text = ",".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_structural_stages(lines, num_stages, seed):
    """Group whole join-graph families into progressively complex stages."""
    graph_to_indices = {}
    for index, line in enumerate(lines):
        signature = graph_signature(line)
        graph_to_indices.setdefault(signature, []).append(index)
    ordered_graphs = sorted(
        graph_to_indices,
        key=lambda graph: (len(graph[0]), len(graph[1]), graph))
    if num_stages > len(ordered_graphs):
        raise ValueError(
            "num_stages={} exceeds {} unique join graphs".format(
                num_stages, len(ordered_graphs)))
    graph_groups = np.array_split(np.asarray(ordered_graphs, dtype=object),
                                  num_stages)
    rng = np.random.default_rng(seed)
    stages = []
    for stage_id, group in enumerate(graph_groups):
        graphs = [tuple(item) for item in group.tolist()]
        indices = [index for graph in graphs
                   for index in graph_to_indices[graph]]
        arrival_order = rng.permutation(indices).tolist()
        stages.append({
            "stage": stage_id,
            "graphs": [repr(graph) for graph in graphs],
            "graph_count": len(graphs),
            "query_count": len(indices),
            "min_tables": min(len(graph_signature(lines[i])[0])
                              for i in indices),
            "max_tables": max(len(graph_signature(lines[i])[0])
                              for i in indices),
            "arrival_indices": arrival_order,
            "arrival_id": stable_hash(arrival_order),
        })
    return stages


def select_adaptation(stage, budget, policy, seed):
    arrivals = stage["arrival_indices"]
    if len(arrivals) <= budget:
        raise ValueError(
            "Stage {} has {} queries, not enough for budget {} plus evaluation"
            .format(stage["stage"], len(arrivals), budget))
    if policy == "first":
        selected = arrivals[:budget]
    elif policy == "random":
        rng = np.random.default_rng(seed + 100003 * (stage["stage"] + 1))
        selected = sorted(rng.choice(arrivals, size=budget,
                                     replace=False).tolist())
    else:
        raise ValueError("Unknown selection policy: {}".format(policy))
    selected_set = set(selected)
    evaluation = [index for index in arrivals if index not in selected_set]
    return selected, evaluation


def select_shifthandler(stage, budget, seed, checkpoint_folder,
                        target_query, target_bitmap, args):
    module_dir = str(Path(args.shift_handler_root).resolve() / "ShiftHandler")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from replay_buffer import summarizer
    arrivals = stage["arrival_indices"]
    features, labels = mscn_run.extract_representation_features(
        args.dataset, target_query, target_bitmap, str(checkpoint_folder),
        arrivals, cuda=args.device, batch_size=args.batch_size,
        num_samples=args.num_samples, alpha=args.alpha,
        contrastive=False, meta=False)
    buffer = summarizer(
        buffer_limit=budget, loss_ada=False,
        concentration=args.shift_concentration,
        is_move=False, seed=seed)
    # Selection happens before executing/labeling the target queries.  CBP
    # therefore receives a neutral label and clusters only model embeddings;
    # passing true target cardinalities here would leak labels outside the
    # adaptation budget through ODMedoids' optional output-gap heuristic.
    for feature, query_id in zip(features, arrivals):
        buffer.process_a_query(feature, 0.0, None, query_id)
    samples, _ = buffer.get_all_samples()
    selected = sorted(int(sample[3]) for sample in samples)
    if len(selected) != budget:
        raise RuntimeError(
            "ShiftHandler selected {} queries, expected {}".format(
                len(selected), budget))
    selected_set = set(selected)
    evaluation = [index for index in arrivals if index not in selected_set]
    return selected, evaluation


def model_path(folder, dataset, alpha, contrastive, meta):
    name = "{}_model_{}".format(dataset, alpha)
    if contrastive:
        name += "_contrastive"
    if meta:
        name += "_meta"
    return str(Path(folder) / (name + ".pkl"))


def load_metrics(path):
    with open(path) as stream:
        return json.load(stream)


def load_execution_times(path, expected_count):
    if path is None:
        return None
    if str(path).endswith(".json"):
        with open(path) as stream:
            payload = json.load(stream)
        raw = payload.get("query_times_seconds", payload)
        return {int(index): float(value) for index, value in raw.items()}
    with open(path) as stream:
        values = [float(line.strip()) for line in stream if line.strip()]
    if len(values) != expected_count:
        raise ValueError(
            "Execution-time trace has {} rows; expected {}".format(
                len(values), expected_count))
    if any(value < 0 for value in values):
        raise ValueError("Execution times must be non-negative")
    return values


def evaluate(dataset, query_file, bitmap_file, checkpoint_folder,
             output_folder, indices, args, contrastive, meta):
    output_folder.mkdir(parents=True, exist_ok=True)
    result_file = output_folder / "predictions.csv"
    mscn_run.predict(
        dataset, query_file, str(result_file), str(checkpoint_folder),
        bitmap_file, cuda=args.device, batch_size=args.batch_size,
        num_samples=args.num_samples, adaptation=False, alpha=args.alpha,
        contrastive=contrastive, meta=meta, include_indices=indices)
    return load_metrics(output_folder / "metrics.json")


def dataset_paths(dataset):
    if dataset == "job-light-ranges":
        return mscn_run.load_job_dataset()
    if dataset == "stats":
        return mscn_run.load_stats_dataset()
    raise ValueError("Sequential driver currently supports job-light-ranges and stats")


def run_one(args, method, policy):
    contrastive, meta = METHODS[method]
    (source_query, min_max, target_query, _result_file, _save_folder,
     _adaptation_query, _validation_file, source_bitmap, target_bitmap,
     _adaptation_bitmap, _tables) = dataset_paths(args.dataset)
    target_lines = read_workload_lines(target_query)
    execution_times = load_execution_times(
        args.target_execution_times, len(target_lines))
    stages = build_structural_stages(target_lines, args.num_stages,
                                     args.stage_seed)
    run_root = (Path(args.output_root).resolve() / args.dataset /
                method / policy / "seed_{:03d}".format(args.seed))
    run_root.mkdir(parents=True, exist_ok=True)
    source_count = len(read_workload_lines(source_query))
    source_rng = np.random.default_rng(args.source_eval_seed)
    source_eval_count = min(args.source_eval_size, source_count)
    source_indices = sorted(source_rng.choice(
        source_count, size=source_eval_count, replace=False).tolist())

    plan = {
        "dataset": args.dataset,
        "method": method,
        "selection_policy": policy,
        "seed": args.seed,
        "stage_seed": args.stage_seed,
        "adaptation_budget_per_stage": args.adaptation_budget,
        "initial_model_folder": (
            None if args.initial_model_folder is None
            else str(Path(args.initial_model_folder).resolve())),
        "target_execution_times": (
            None if args.target_execution_times is None
            else str(Path(args.target_execution_times).resolve())),
        "stages": stages,
        "source_evaluation_indices": source_indices,
        "source_evaluation_id": stable_hash(source_indices),
        "training": {
            "epochs_per_stage": args.epochs,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "learning_rate": args.lr,
            "target_repeat": args.repeat,
            "num_materialized_samples": args.num_samples,
            "device": args.device,
        },
    }
    (run_root / "plan.json").write_text(json.dumps(
        plan, indent=2, sort_keys=True))

    previous_folder = (None if args.initial_model_folder is None
                       else Path(args.initial_model_folder).resolve())
    previous_checkpoint = (
        None if previous_folder is None else model_path(
            previous_folder, args.dataset, args.alpha, contrastive, meta))
    if previous_checkpoint is not None and not Path(previous_checkpoint).exists():
        raise FileNotFoundError(
            "Initial historical-source checkpoint not found: {}".format(
                previous_checkpoint))
    if method == "shifthandler" and previous_folder is None:
        raise ValueError("ShiftHandler requires --initial_model_folder")
    cumulative_adaptation_seconds = 0.0
    cumulative_query_execution_seconds = 0.0
    timeline = []
    for stage in stages:
        stage_id = stage["stage"]
        if method == "shifthandler":
            adaptation_indices, evaluation_indices = select_shifthandler(
                stage, args.adaptation_budget, args.seed, previous_folder,
                target_query, target_bitmap, args)
        else:
            adaptation_indices, evaluation_indices = select_adaptation(
                stage, args.adaptation_budget, policy, args.seed)
        stage_folder = run_root / "stage_{:02d}".format(stage_id)
        stage_folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            **{key: value for key, value in stage.items()
               if key != "arrival_indices"},
            "adaptation_indices": adaptation_indices,
            "adaptation_id": stable_hash(adaptation_indices),
            "evaluation_indices": evaluation_indices,
            "evaluation_id": stable_hash(evaluation_indices),
            "warm_start_checkpoint": previous_checkpoint,
        }
        (stage_folder / "manifest.json").write_text(json.dumps(
            manifest, indent=2, sort_keys=True))

        pre_metrics = None
        if previous_folder is not None:
            pre_metrics = evaluate(
                args.dataset, target_query, target_bitmap, previous_folder,
                stage_folder / "pre_adaptation", evaluation_indices, args,
                contrastive, meta)

        start = time.perf_counter()
        mscn_run.train_triplet(
            args.dataset, source_query, min_max, str(stage_folder),
            source_bitmap, target_query, target_bitmap, 0, 0,
            num_samples=args.num_samples, cuda=args.device,
            batch_size=args.batch_size, num_epochs=args.epochs,
            alpha=args.alpha, lr=args.lr, repeat=args.repeat,
            contrastive=contrastive, meta=meta,
            adaptation_indices=adaptation_indices,
            initial_model_file=previous_checkpoint)
        adaptation_seconds = time.perf_counter() - start
        cumulative_adaptation_seconds += adaptation_seconds
        if execution_times is None:
            query_execution_seconds = None
        else:
            missing_times = [index for index in adaptation_indices
                             if index not in execution_times]
            if missing_times:
                raise ValueError(
                    "Execution-time trace lacks selected query IDs: {}".format(
                        missing_times))
            query_execution_seconds = sum(
                execution_times[index] for index in adaptation_indices)
        if query_execution_seconds is not None:
            cumulative_query_execution_seconds += query_execution_seconds
        checkpoint = model_path(stage_folder, args.dataset, args.alpha,
                                contrastive, meta)

        post_metrics = evaluate(
            args.dataset, target_query, target_bitmap, stage_folder,
            stage_folder / "post_adaptation", evaluation_indices, args,
            contrastive, meta)
        source_metrics = evaluate(
            args.dataset, source_query, source_bitmap, stage_folder,
            stage_folder / "source_replay", source_indices, args,
            contrastive, meta)
        record = {
            "stage": stage_id,
            "method": method,
            "selection_policy": policy,
            "adaptation_queries": len(adaptation_indices),
            "evaluation_queries": len(evaluation_indices),
            "cumulative_labeled_target_queries": (
                (stage_id + 1) * args.adaptation_budget),
            "adaptation_seconds": adaptation_seconds,
            "cumulative_adaptation_seconds": cumulative_adaptation_seconds,
            "query_execution_seconds": query_execution_seconds,
            "cumulative_query_execution_seconds": (
                None if execution_times is None
                else cumulative_query_execution_seconds),
            "pre_adaptation": pre_metrics,
            "post_adaptation": post_metrics,
            "source_replay": source_metrics,
        }
        timeline.append(record)
        (stage_folder / "stage_metrics.json").write_text(json.dumps(
            record, indent=2, sort_keys=True))
        previous_checkpoint = checkpoint
        previous_folder = stage_folder

    (run_root / "timeline.json").write_text(json.dumps(
        timeline, indent=2, sort_keys=True))
    rows = []
    for record in timeline:
        row = {
            "stage": record["stage"],
            "method": method,
            "selection_policy": policy,
            "adaptation_seconds": record["adaptation_seconds"],
            "cumulative_adaptation_seconds": record[
                "cumulative_adaptation_seconds"],
            "cumulative_labeled_target_queries": record[
                "cumulative_labeled_target_queries"],
            "query_execution_seconds": (
                "" if record["query_execution_seconds"] is None
                else record["query_execution_seconds"]),
            "cumulative_query_execution_seconds": (
                "" if record["cumulative_query_execution_seconds"] is None
                else record["cumulative_query_execution_seconds"]),
        }
        for prefix in ("pre_adaptation", "post_adaptation", "source_replay"):
            metrics = record[prefix]
            for metric in ("median_qerror", "p95_qerror", "p99_qerror"):
                row["{}_{}".format(prefix, metric)] = (
                    "" if metrics is None else metrics[metric])
        rows.append(row)
    with (run_root / "timeline.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="job-light-ranges",
                        choices=["job-light-ranges", "stats"])
    parser.add_argument("--methods", default="adapt,cmda")
    parser.add_argument("--selection_policies", default="first,random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage_seed", type=int, default=20260928)
    parser.add_argument("--source_eval_seed", type=int, default=20260929)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--adaptation_budget", type=int, default=10)
    parser.add_argument("--source_eval_size", type=int, default=500)
    parser.add_argument(
        "--initial_model_folder",
        help=("folder containing the method-specific checkpoint and structure "
              "trained only on the historical source workload"))
    parser.add_argument(
        "--target_execution_times",
        help=("optional line-aligned target trace, one execution time in "
              "seconds per target query"))
    parser.add_argument("--shift_handler_root",
                        default=os.environ.get("CMDA_SHIFTHANDLER_ROOT", str(Path(__file__).resolve().parents[1] / "third_party/ShiftHandler")))
    parser.add_argument("--shift_concentration", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--output_root", default="../results/sequential_drift")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.adaptation_budget <= 0:
        parser.error("--adaptation_budget must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    mscn_run.set_global_seed(args.seed)
    methods = parse_list(args.methods)
    policies = parse_list(args.selection_policies)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError("Unsupported methods {}".format(unknown))
    output_roots = []
    for method in methods:
        method_policies = ["cbp"] if method == "shifthandler" else policies
        for policy in method_policies:
            # Reset stochastic training state so paired policies/methods are
            # auditable from the requested seed.
            mscn_run.set_global_seed(args.seed)
            output_roots.append(str(run_one(args, method, policy)))
    print("Completed sequential runs:")
    for output_root in output_roots:
        print(output_root)


if __name__ == "__main__":
    main()
