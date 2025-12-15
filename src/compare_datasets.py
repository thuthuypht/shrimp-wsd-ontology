#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare performance of LR, SVM, CNN and Hybrid on:

- Dataset 1 (original, 233 farms)
- Dataset 2 (expanded, ~1500 records)

Each dataset is evaluated over multiple random seeds (default 5),
using the run_experiment() function from wsd_ontology_ensemble.py.
"""

import argparse
import numpy as np

from wsd_ontology_ensemble import run_experiment


METHODS = ["lr", "svm", "cnn", "hybrid"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def evaluate_dataset(data_path: str, seeds):
    """Run multiple experiments and aggregate results."""
    results_all = {
        method: {metric: [] for metric in METRICS} for method in METHODS
    }

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Running dataset {data_path} with random_state = {seed}")
        print("=" * 70)

        res = run_experiment(
            data_path=data_path,
            ontology_path=None,       # skip ontology writing for repeated runs
            output_owl_path=None,
            threshold=0.5,
            random_state=seed,
        )

        for method in METHODS:
            for metric in METRICS:
                results_all[method][metric].append(res[method][metric])

    # Compute mean and std
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in METRICS:
            arr = np.array(results_all[method][metric])
            summary[method][metric + "_mean"] = arr.mean()
            summary[method][metric + "_std"] = arr.std(ddof=1)
    return summary


def print_summary(summary, dataset_name: str):
    """Pretty-print mean ± std for each method."""
    print("\n" + "#" * 80)
    print(f"Summary over runs for {dataset_name}")
    print("#" * 80)
    header = "{:<10s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "Method", "Acc", "Prec", "Recall", "F1", "AUC"
    )
    print(header)
    print("-" * len(header))

    for method in METHODS:
        m = summary[method]
        def fmt(metric):
            return f"{m[metric + '_mean']:.3f}±{m[metric + '_std']:.3f}"

        line = "{:<10s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
            method,
            fmt("accuracy"),
            fmt("precision"),
            fmt("recall"),
            fmt("f1"),
            fmt("auc"),
        )
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Dataset 1 (original) and Dataset 2 (expanded)."
    )
    parser.add_argument(
        "--orig_data_path",
        type=str,
        required=True,
        help="Path to original dataset (Dataset 1), e.g. data/WSSVRiskFactor_rev.xlsx",
    )
    parser.add_argument(
        "--expanded_data_path",
        type=str,
        required=True,
        help="Path to expanded dataset (Dataset 2), e.g. data/WSSVRiskFactor_expanded.csv",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for repeated experiments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary_orig = evaluate_dataset(args.orig_data_path, args.seeds)
    summary_exp = evaluate_dataset(args.expanded_data_path, args.seeds)

    print_summary(summary_orig, "Dataset 1 (Original)")
    print_summary(summary_exp, "Dataset 2 (Expanded)")


if __name__ == "__main__":
    main()
