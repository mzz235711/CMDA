"""Sequential/repeated workload-drift evaluation for Bao cost estimation."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
import sys

import numpy as np

import cost_experiment
import dataset
import train


METHODS = {
    "adapt": {"contrastive": False, "meta": False},
    "cmda": {"contrastive": True, "meta": True},
    "shifthandler": {"contrastive": False, "meta": False},
}


def ids_hash(ids):
    return hashlib.sha256(
        ",".join(str(value) for value in ids).encode("utf-8")).hexdigest()


def read_sql(path):
    with open(path) as stream:
        return [line.strip() for line in stream if line.strip()]


def graph_signature(sql):
    normalized = " ".join(sql.lower().rstrip(";").split())
    match = re.search(r"\bfrom\b(.*?)(?:\bwhere\b(.*))?$", normalized)
    if not match:
        raise ValueError("Cannot parse SQL: {}".format(sql[:160]))
    tables = tuple(sorted(part.strip() for part in match.group(1).split(",")))
    joins = []
    for condition in re.split(r"\s+and\s+", match.group(2) or ""):
        comparison = re.match(
            r"(.+?)\s*(<=|>=|<>|!=|=|<|>)\s*(.+)$", condition.strip())
        if not comparison:
            continue
        left, operator, right = (value.strip() for value in comparison.groups())
        if re.fullmatch(r"[a-z_][\w$]*\.[a-z_][\w$]*", right):
            endpoints = sorted((left, right))
            joins.append(endpoints[0] + operator + endpoints[1])
    return tables, tuple(sorted(joins))


def build_stages(sql_lines, available_ids, count, seed):
    grouped = {}
    for plan_id in available_ids:
        grouped.setdefault(graph_signature(sql_lines[plan_id]), []).append(plan_id)
    graphs = sorted(grouped, key=lambda value: (len(value[0]), len(value[1]), value))
    if count > len(graphs):
        raise ValueError("Not enough distinct join graphs for {} stages".format(count))
    groups = np.array_split(np.asarray(graphs, dtype=object), count)
    rng = np.random.default_rng(seed)
    stages = []
    for stage_id, group in enumerate(groups):
        stage_graphs = [tuple(value) for value in group.tolist()]
        ids = [plan_id for graph in stage_graphs for plan_id in grouped[graph]]
        arrivals = rng.permutation(ids).tolist()
        stages.append({
            "stage": stage_id,
            "graph_count": len(stage_graphs),
            "query_count": len(ids),
            "graphs": [repr(value) for value in stage_graphs],
            "arrival_indices": arrivals,
            "arrival_id": ids_hash(arrivals),
        })
    return stages


def select(stage, budget, policy, seed):
    arrivals = stage["arrival_indices"]
    if len(arrivals) <= budget:
        raise ValueError("Stage {} is too small".format(stage["stage"]))
    if policy == "first":
        adaptation = arrivals[:budget]
    elif policy == "random":
        rng = np.random.default_rng(seed + 100003 * (stage["stage"] + 1))
        adaptation = sorted(rng.choice(arrivals, budget, replace=False).tolist())
    else:
        raise ValueError(policy)
    selected = set(adaptation)
    evaluation = [value for value in arrivals if value not in selected]
    return adaptation, evaluation


def select_shifthandler_aligned(stage, budget, seed, reg, plan_dir,
                                label_file, shift_handler_root, concentration):
    module_dir = str(Path(shift_handler_root).resolve() / "ShiftHandler")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from replay_buffer import summarizer
    arrivals = stage["arrival_indices"]
    experiences, labels, ids, _ = train.load_aligned_experience(
        plan_dir, label_file, arrivals)
    plans = [value[0] for value in experiences]
    features = reg.representation_features(plans)
    buffer = summarizer(
        buffer_limit=budget, loss_ada=False,
        concentration=concentration, is_move=False, seed=seed)
    for feature, label, plan, plan_id in zip(features, labels, plans, ids):
        buffer.process_a_query(feature, label, plan, plan_id)
    samples, _ = buffer.get_all_samples()
    selected = sorted(int(sample[3]) for sample in samples)
    if len(selected) != budget:
        raise RuntimeError(
            "ShiftHandler selected {} queries, expected {}".format(
                len(selected), budget))
    selected_set = set(selected)
    evaluation = [value for value in arrivals if value not in selected_set]
    return selected, evaluation


def metrics(reg, plan_dir, label_file, indices, output_dir):
    experiences, labels, ids, alignment = train.load_aligned_experience(
        plan_dir, label_file, indices)
    predictions = [value[0] for value in reg.predict([value[0] for value in experiences])]
    errors = [max(prediction / label, label / prediction)
              for prediction, label in zip(predictions, labels)]
    result = {
        "evaluation_count": len(errors),
        "median_qerror": float(np.median(errors)),
        "p95_qerror": float(np.percentile(errors, 95)),
        "p99_qerror": float(np.percentile(errors, 99)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    with (output_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["plan_id", "label", "prediction", "qerror"])
        writer.writerows(zip(ids, labels, predictions, errors))
    (output_dir / "alignment.json").write_text(json.dumps(alignment, indent=2, sort_keys=True))
    return result


def load_saved(folder, args, method):
    import model
    flags = METHODS[method]
    reg = model.BaoRegression(
        have_cache_data=True, alpha=args.alpha, lr=args.lr,
        contrastive=flags["contrastive"], meta=flags["meta"])
    reg.load(str(folder / "model"), args.repeat)
    return reg


def run_one(args, method, policy):
    flags = METHODS[method]
    config = dataset.load_multiseed_dataset(args.dataset)
    _, target_labels, target_ids, _ = train.load_aligned_experience(
        config["adaptation_plans"], config["adaptation_labels"])
    sql_lines = read_sql(config["adaptation_sql"])
    if max(target_ids) >= len(sql_lines):
        raise ValueError("Adaptation SQL does not cover all plan IDs")
    stages = build_stages(sql_lines, target_ids, args.num_stages, args.stage_seed)
    root = (Path(args.output_root).resolve() / args.dataset / method / policy /
            "seed_{:03d}".format(args.seed))
    root.mkdir(parents=True, exist_ok=True)
    source_ids = train.load_aligned_experience(
        config["train_plans"], config["train_labels"])[2]
    source_rng = np.random.default_rng(args.source_eval_seed)
    source_eval = sorted(source_rng.choice(
        source_ids, min(args.source_eval_size, len(source_ids)),
        replace=False).tolist())
    previous = (None if args.initial_model_folder is None else
                Path(args.initial_model_folder).resolve())
    if method == "shifthandler" and previous is None:
        raise ValueError("ShiftHandler requires --initial_model_folder")
    timeline = []
    cumulative_adaptation = 0.0
    cumulative_execution = 0.0
    (root / "plan.json").write_text(json.dumps({
        "dataset": args.dataset, "method": method, "policy": policy,
        "seed": args.seed, "stage_seed": args.stage_seed,
        "adaptation_budget": args.adaptation_budget,
        "initial_model_folder": None if previous is None else str(previous),
        "stages": stages, "source_evaluation_indices": source_eval,
        "source_evaluation_id": ids_hash(source_eval),
    }, indent=2, sort_keys=True))
    for stage in stages:
        if method == "shifthandler":
            selection_reg = load_saved(previous, args, method)
            adaptation, evaluation = select_shifthandler_aligned(
                stage, args.adaptation_budget, args.seed, selection_reg,
                config["adaptation_plans"], config["adaptation_labels"],
                args.shift_handler_root, args.shift_concentration)
        else:
            adaptation, evaluation = select(
                stage, args.adaptation_budget, policy, args.seed)
        folder = root / "stage_{:02d}".format(stage["stage"])
        folder.mkdir(parents=True, exist_ok=True)
        pre = None
        if previous is not None:
            pre_reg = load_saved(previous, args, method)
            pre = metrics(pre_reg, config["adaptation_plans"],
                          config["adaptation_labels"], evaluation,
                          folder / "pre_adaptation")
        started = time.perf_counter()
        reg, alignment = train.train_and_save_model_from_file(
            str(folder), config["train_plans"], config["adaptation_plans"],
            config["train_labels"], config["adaptation_labels"],
            alpha=args.alpha, repeat=args.repeat,
            contrastive=flags["contrastive"], meta=flags["meta"], lr=args.lr,
            adaptation_indices=adaptation, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed,
            initial_model_folder=None if previous is None else str(previous))
        seconds = time.perf_counter() - started
        cumulative_adaptation += seconds
        execution_seconds = sum(target_labels[target_ids.index(i)] for i in adaptation)
        cumulative_execution += execution_seconds
        post = metrics(reg, config["adaptation_plans"],
                       config["adaptation_labels"], evaluation,
                       folder / "post_adaptation")
        source = metrics(reg, config["train_plans"], config["train_labels"],
                         source_eval, folder / "source_replay")
        record = {
            "stage": stage["stage"], "adaptation_indices": adaptation,
            "adaptation_id": ids_hash(adaptation),
            "evaluation_indices": evaluation,
            "evaluation_id": ids_hash(evaluation),
            "warm_start_folder": None if previous is None else str(previous),
            "adaptation_seconds": seconds,
            "cumulative_adaptation_seconds": cumulative_adaptation,
            "query_execution_seconds": execution_seconds,
            "cumulative_query_execution_seconds": cumulative_execution,
            "pre_adaptation": pre, "post_adaptation": post,
            "source_replay": source, "training_alignment": alignment,
        }
        (folder / "stage_metrics.json").write_text(
            json.dumps(record, indent=2, sort_keys=True))
        timeline.append(record)
        previous = folder
    (root / "timeline.json").write_text(json.dumps(timeline, indent=2, sort_keys=True))
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["job-light-ranges", "stats"],
                        default="job-light-ranges")
    parser.add_argument("--methods", default="adapt,cmda")
    parser.add_argument("--policies", default="first,random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage_seed", type=int, default=20260928)
    parser.add_argument("--source_eval_seed", type=int, default=20260929)
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--adaptation_budget", type=int, default=10)
    parser.add_argument("--source_eval_size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--initial_model_folder")
    parser.add_argument("--shift_handler_root",
                        default=os.environ.get("CMDA_SHIFTHANDLER_ROOT", str(Path(__file__).resolve().parents[2] / "third_party/ShiftHandler")))
    parser.add_argument("--shift_concentration", type=float, default=1.0)
    parser.add_argument("--output_root", default="../../results/cost_sequential_drift")
    args = parser.parse_args()
    if args.epochs <= 0 or args.adaptation_budget <= 0:
        parser.error("epochs and adaptation_budget must be positive")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        parser.error("unsupported methods: {}".format(unknown))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for method in methods:
        method_policies = ["cbp"] if method == "shifthandler" else policies
        for policy in method_policies:
            cost_experiment.set_global_seed(args.seed)
            print("Completed:", run_one(args, method, policy))


if __name__ == "__main__":
    main()
