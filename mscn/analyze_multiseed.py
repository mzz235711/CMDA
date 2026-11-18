"""Validate and summarize paired multi-seed q-error experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = [
    "mean_qerror", "median_qerror", "p90_qerror", "p95_qerror",
    "p99_qerror", "max_qerror",
]


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    standard_error = stats.sem(values)
    delta = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * standard_error
    return mean, float(mean - delta), float(mean + delta)


def holm_adjust(pvalues):
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="../results/cardinality_multiseed")
    parser.add_argument("--datasets", default="job-light-ranges,stats")
    parser.add_argument("--methods", default="adapt,cmda_c,cmda_m,cmda")
    parser.add_argument(
        "--comparison_pairs",
        default="adapt:cmda_c,adapt:cmda_m,adapt:cmda,cmda_c:cmda,cmda_m:cmda",
        help="comma-separated paired comparisons in baseline:method form")
    parser.add_argument("--adaptation_budget", type=int, default=10)
    parser.add_argument("--expected_seeds", type=int, default=20)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.results_root)
    if not root.is_absolute():
        root = (script_dir / root).resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    comparison_pairs = []
    for item in args.comparison_pairs.split(","):
        baseline, method = (x.strip() for x in item.split(":", 1))
        if baseline not in methods or method not in methods:
            raise ValueError("Comparison uses a method not listed in --methods")
        comparison_pairs.append((baseline, method))
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset in datasets:
        reference_by_seed = {}
        fixed_evaluation_ids = set()
        for method in methods:
            method_dir = root / dataset / "budget_{}".format(
                args.adaptation_budget) / method
            for seed_dir in sorted(method_dir.glob("seed_*")):
                manifest_file = seed_dir / "manifest.json"
                metrics_file = seed_dir / "metrics.json"
                if not manifest_file.exists() or not metrics_file.exists():
                    continue
                manifest = json.loads(manifest_file.read_text())
                metrics = json.loads(metrics_file.read_text())
                seed = int(manifest["seed"])
                pairing = (manifest["selection_id"], manifest["evaluation_id"])
                if seed in reference_by_seed and reference_by_seed[seed] != pairing:
                    raise RuntimeError(
                        "Unpaired selection/evaluation set for {} seed {}".format(
                            dataset, seed))
                reference_by_seed[seed] = pairing
                fixed_evaluation_ids.add(manifest["evaluation_id"])
                row = {"dataset": dataset, "method": method, "seed": seed,
                       **{key: metrics[key] for key in METRICS}}
                records.append(row)
        if len(fixed_evaluation_ids) != 1:
            raise RuntimeError(
                "Evaluation workload changes across seeds for {}".format(dataset))
        for method in methods:
            count = sum(1 for row in records
                        if row["dataset"] == dataset and row["method"] == method)
            if count != args.expected_seeds:
                raise RuntimeError(
                    "Expected {} seeds for {}/{}, found {}".format(
                        args.expected_seeds, dataset, method, count))

    raw_file = output_dir / "per_seed_metrics.csv"
    with raw_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "method", "seed"] + METRICS)
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for dataset in datasets:
        for method in methods:
            selected = [r for r in records
                        if r["dataset"] == dataset and r["method"] == method]
            for metric in METRICS:
                values = np.asarray([r[metric] for r in selected], dtype=float)
                mean, ci_low, ci_high = mean_ci(values)
                summaries.append({
                    "dataset": dataset, "method": method, "metric": metric,
                    "n": len(values), "mean": mean,
                    "std": float(np.std(values, ddof=1)),
                    "ci95_low": ci_low, "ci95_high": ci_high,
                })
    summary_file = output_dir / "summary.csv"
    with summary_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    comparisons = []
    for dataset in datasets:
        family = []
        for baseline, method in comparison_pairs:
            for metric in METRICS:
                baseline_rows = sorted(
                    (r for r in records if r["dataset"] == dataset and
                     r["method"] == baseline), key=lambda r: r["seed"])
                method_rows = sorted(
                    (r for r in records if r["dataset"] == dataset and
                     r["method"] == method), key=lambda r: r["seed"])
                if [r["seed"] for r in baseline_rows] != [r["seed"] for r in method_rows]:
                    raise RuntimeError("Seed mismatch in paired comparison")
                baseline_values = np.asarray([r[metric] for r in baseline_rows])
                method_values = np.asarray([r[metric] for r in method_rows])
                improvement = baseline_values - method_values
                try:
                    test = stats.wilcoxon(
                        baseline_values, method_values,
                        alternative="two-sided", zero_method="wilcox")
                    pvalue = float(test.pvalue)
                    statistic = float(test.statistic)
                except ValueError:
                    pvalue, statistic = 1.0, 0.0
                relative = np.divide(
                    improvement, baseline_values,
                    out=np.zeros_like(improvement, dtype=float),
                    where=baseline_values != 0)
                family.append({
                    "dataset": dataset, "baseline": baseline,
                    "method": method, "metric": metric,
                    "wilcoxon_statistic": statistic,
                    "p_raw": pvalue,
                    "median_relative_improvement": float(np.median(relative)),
                    "rank_biserial": float(rank_biserial(improvement)),
                })
        adjusted = holm_adjust([row["p_raw"] for row in family])
        for row, p_adjusted in zip(family, adjusted):
            row["p_holm"] = p_adjusted
            row["significant_0.05"] = p_adjusted < 0.05
            comparisons.append(row)
    comparison_file = output_dir / "paired_wilcoxon_holm.csv"
    with comparison_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    print("Wrote", raw_file)
    print("Wrote", summary_file)
    print("Wrote", comparison_file)


if __name__ == "__main__":
    main()
