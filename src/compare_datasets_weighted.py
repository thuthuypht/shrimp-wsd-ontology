#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare LR, SVM, CNN, ontology-only, and Hybrid (with tuned ontology risk)
on:

- Dataset 1 (original, 233 farms)
- Dataset 2 (expanded, ~1500 records)

The ensemble weight search is constrained so that w4 (ontology) >= min_w4,
so the final hybrid ALWAYS uses ontology.
"""

import argparse
import numpy as np

from wsd_ontology_ensemble_weighted import run_experiment

METHODS = ["ont", "lr", "svm", "cnn", "hybrid"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def evaluate_dataset(data_path: str, seeds, min_w4: float):
    results_all = {
        method: {metric: [] for metric in METRICS} for method in METHODS
    }

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Running dataset {data_path} with random_state = {seed}")
        print("=" * 70)

        res = run_experiment(
            data_path=data_path,
            ontology_path=None,        # skip OWL writing in repeated runs
            output_owl_path=None,
            threshold=0.5,
            random_state=seed,
            min_w4=min_w4,
        )

        for method in METHODS:
            for metric in METRICS:
                results_all[method][metric].append(res[method][metric])

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in METRICS:
            arr = np.array(results_all[method][metric])
            summary[method][metric + "_mean"] = arr.mean()
            summary[method][metric + "_std"] = arr.std(ddof=1)
    return summary


def print_summary(summary, dataset_name: str, note: str = ""):
    print("\n" + "#" * 80)
    title = f"Summary over runs for {dataset_name}"
    if note:
        title += f" ({note})"
    print(title)
    print("#" * 80)

    header = "{:<8s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "Method", "Acc", "Prec", "Recall", "F1", "AUC"
    )
    print(header)
    print("-" * len(header))

    for method in METHODS:
        m = summary[method]

        def fmt(metric):
            return f"{m[metric + '_mean']:.3f}±{m[metric + '_std']:.3f}"

        line = "{:<8s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
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
        description="Compare Dataset 1 and Dataset 2 with tuned ontology and w4>=min_w4."
    )
    parser.add_argument(
        "--orig_data_path",
        type=str,
        required=True,
        help="Path to original dataset (Dataset 1).",
    )
    parser.add_argument(
        "--expanded_data_path",
        type=str,
        required=True,
        help="Path to expanded dataset (Dataset 2).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds for repeated experiments.",
    )
    parser.add_argument(
        "--min_w4",
        type=float,
        default=0.1,
        help="Minimum weight for ontology risk in the ensemble.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    summary_orig = evaluate_dataset(args.orig_data_path, args.seeds, args.min_w4)
    summary_exp = evaluate_dataset(args.expanded_data_path, args.seeds, args.min_w4)

    note = f"tuned ontology, w4>={args.min_w4:.1f}"
    print_summary(summary_orig, "Dataset 1 (Original)", note)
    print_summary(summary_exp, "Dataset 2 (Expanded)", note)


if __name__ == "__main__":
    main()
