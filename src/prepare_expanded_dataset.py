#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare Dataset 2 (expanded) from the original WSSVRiskFactor_rev.xlsx.

Steps (as described in the paper):

1. Random oversampling of the minority class (VirusDetected = 1)
   to match the majority-class size.

2. Gaussian noise augmentation applied only to selected numeric
   features (e.g., Temperature, Salinity, StockingDensity_PL/40MeterSquare)
   with small standard deviation sigma = 0.04 on min-max normalised data,
   then mapped back to original scale.

The script saves an expanded dataset of approximately target_size rows
(default ~1500 records).
"""

import argparse
import numpy as np
import pandas as pd
import os

# Column names in your dataset
TARGET_COL = "VirusDetected"

# Numeric features to augment with Gaussian noise
AUG_NUM_COLS = [
    "Temperature",
    "Salinity",
    "StockingDensity_PL/40MeterSquare",
    "PreviousPrevalence(%)",
    # Add more continuous columns here if desired
]


def load_original_dataset(path: str) -> pd.DataFrame:
    """Load the original dataset from Excel/CSV."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Basic sanity check
    missing = [c for c in [TARGET_COL] + AUG_NUM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in file {path}: {missing}")

    # Drop rows with missing key values
    df = df.dropna(subset=[TARGET_COL] + AUG_NUM_COLS)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def oversample_minority(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Random oversampling of minority class to match majority size."""
    counts = df[TARGET_COL].value_counts()
    if len(counts) != 2:
        raise ValueError("Expected a binary target column with two classes.")

    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    df_major = df[df[TARGET_COL] == majority_label]
    df_minor = df[df[TARGET_COL] == minority_label]

    n_major = len(df_major)
    n_minor = len(df_minor)

    df_minor_oversampled = df_minor.sample(
        n=n_major, replace=True, random_state=random_state
    )

    df_balanced = pd.concat([df_major, df_minor_oversampled], ignore_index=True)
    df_balanced = df_balanced.sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    print("Original class counts:", counts.to_dict())
    print("Balanced class counts:", df_balanced[TARGET_COL].value_counts().to_dict())
    print("Balanced size:", len(df_balanced))
    return df_balanced


def add_gaussian_noise(
    df: pd.DataFrame,
    base_df: pd.DataFrame,
    sigma: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add Gaussian noise N(0, sigma^2) to selected numeric columns.

    We first min-max normalise each column to [0,1] using base_df stats,
    add noise, clip to [0,1], then map back to the original scale.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()

    for col in AUG_NUM_COLS:
        if col not in df_noisy.columns:
            continue

        col_vals = df_noisy[col].astype(float).values
        col_min = base_df[col].min()
        col_max = base_df[col].max()

        if col_max <= col_min:
            # Constant column, skip
            continue

        norm = (col_vals - col_min) / (col_max - col_min)
        noise = rng.normal(loc=0.0, scale=sigma, size=len(df_noisy))
        norm_noisy = np.clip(norm + noise, 0.0, 1.0)
        col_noisy = norm_noisy * (col_max - col_min) + col_min
        df_noisy[col] = col_noisy

    return df_noisy


def create_expanded_dataset(
    input_path: str,
    output_path: str,
    target_size: int = 1500,
    sigma: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create Dataset 2 (expanded) according to the 2-step procedure and
    save it to output_path.
    """
    df_orig = load_original_dataset(input_path)
    df_balanced = oversample_minority(df_orig, random_state=random_state)

    n_bal = len(df_balanced)
    if n_bal == 0:
        raise ValueError("Balanced dataset is empty.")

    # Number of full noisy copies to create in addition to the original
    n_rounds = max(int(target_size // n_bal) - 1, 0)

    augmented_list = [df_balanced]  # original balanced data (no noise)

    # Full noisy copies
    for r in range(n_rounds):
        df_noisy = add_gaussian_noise(
            df_balanced,
            base_df=df_balanced,
            sigma=sigma,
            random_state=random_state + r + 1,
        )
        augmented_list.append(df_noisy)

    df_expanded = pd.concat(augmented_list, ignore_index=True)

    # If still below target_size, create extra noisy samples by sampling rows
    if len(df_expanded) < target_size:
        remaining = target_size - len(df_expanded)
        df_sample_base = df_balanced.sample(
            n=remaining, replace=True, random_state=random_state + 999
        )
        df_sample_noisy = add_gaussian_noise(
            df_sample_base,
            base_df=df_balanced,
            sigma=sigma,
            random_state=random_state + 1000,
        )
        df_expanded = pd.concat([df_expanded, df_sample_noisy], ignore_index=True)

    # If above target_size, subsample to exact size
    if len(df_expanded) > target_size:
        df_expanded = df_expanded.sample(
            n=target_size, random_state=random_state + 2025
        ).reset_index(drop=True)

    print("\nExpanded dataset size:", len(df_expanded))
    print("Class counts (expanded):", df_expanded[TARGET_COL].value_counts().to_dict())

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.lower().endswith(".csv"):
        df_expanded.to_csv(output_path, index=False)
    else:
        df_expanded.to_excel(output_path, index=False)

    print("Expanded dataset saved to:", output_path)
    return df_expanded


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an expanded WSD dataset (~1500 rows) from the original file."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to original WSSVRiskFactor_rev.xlsx (Dataset 1).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save expanded dataset (Dataset 2), e.g. data/WSSVRiskFactor_expanded.csv",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=1500,
        help="Target number of records for the expanded dataset (approx).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.04,
        help="Standard deviation for Gaussian noise on normalised features.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_expanded_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        target_size=args.target_size,
        sigma=args.sigma,
        random_state=42,
    )


if __name__ == "__main__":
    main()
