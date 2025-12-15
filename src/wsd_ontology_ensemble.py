#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ontology-guided ensemble learning for White Spot Disease (WSD) diagnosis.

Implements Algorithm 1 with grid search over ensemble weights on a given dataset.

Author: <your name>
"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from owlready2 import get_ontology, default_world


# Columns (same as before)
TARGET_COL = "VirusDetected"
TEMPERATURE_COL = "Temperature"
SALINITY_COL = "Salinity"
STOCKING_COL = "StockingDensity_PL/40MeterSquare"
PREV_PREVALENCE_COL = "PreviousPrevalence(%)"

FEATURE_COLS = [
    TEMPERATURE_COL,
    SALINITY_COL,
    STOCKING_COL,
    PREV_PREVALENCE_COL,
]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_dataset(path: str):
    """Load dataset from Excel/CSV and return df, X, y, feature_names."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in file {path}: {missing}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    return df, X, y, FEATURE_COLS


# ----------------------------------------------------------------------
# CNN model
# ----------------------------------------------------------------------

def build_cnn(input_dim: int):
    """Build a simple 1D-CNN for WSD prediction from tabular data."""
    model = Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(input_dim, 1),
        )
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_binary_classifier(y_true, y_prob, threshold=0.5, description="model"):
    """Compute accuracy, precision, recall, F1 and ROC-AUC."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\n=== Results for {description} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


# ----------------------------------------------------------------------
# Ontology-based risk score
# ----------------------------------------------------------------------

def ontology_risk_score(row):
    """
    Simple ontology-inspired risk score r_ont in [0, 1].
    Adjust thresholds to match SWRL rules if needed.
    """
    score = 0.0
    if row[TEMPERATURE_COL] > 32:
        score += 1.0
    if row[SALINITY_COL] < 5:
        score += 1.0
    if row[STOCKING_COL] > 25:
        score += 1.0
    if row[PREV_PREVALENCE_COL] > 0:
        score += 1.0

    return score / 4.0


# ----------------------------------------------------------------------
# Weight search on validation set
# ----------------------------------------------------------------------

def search_best_weights(
    y_val,
    p_lr_val,
    p_svm_val,
    p_cnn_val,
    r_ont_val,
    threshold=0.5,
):
    """
    Full grid search over weights (w1, w2, w3, w4) in {0,0.1,...,1}
    with sum = 1. Returns weights that maximise F1 on validation,
    tie-breaking by ROC-AUC.
    """
    best_w = None
    best_f1 = -1.0
    best_auc = -1.0

    for i in range(11):
        for j in range(11 - i):
            for k in range(11 - i - j):
                l = 10 - i - j - k
                w = np.array([i, j, k, l], dtype=float) / 10.0

                R_val = (
                    w[0] * p_lr_val
                    + w[1] * p_svm_val
                    + w[2] * p_cnn_val
                    + w[3] * r_ont_val
                )
                y_pred_val = (R_val >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred_val, zero_division=0)
                try:
                    auc = roc_auc_score(y_val, R_val)
                except ValueError:
                    auc = float("nan")

                if f1 > best_f1 or (np.isclose(f1, best_f1) and auc > best_auc):
                    best_f1 = f1
                    best_auc = auc
                    best_w = w

    print("\nBest weights on validation (w1, w2, w3, w4) = "
          f"{tuple(best_w)} with F1 = {best_f1:.4f}, AUC = {best_auc:.4f}")
    print("  w1 = LR,  w2 = SVM,  w3 = CNN,  w4 = ontology-risk")
    return best_w


# ----------------------------------------------------------------------
# Ontology writing (optional)
# ----------------------------------------------------------------------

def attach_results_to_ontology(
    base_owl_path: str,
    output_owl_path: str,
    df_all: pd.DataFrame,
    test_indices: np.ndarray,
    hybrid_scores: np.ndarray,
    hybrid_preds: np.ndarray,
):
    """Attach predictions to ontology individuals and save a new OWL file."""
    print("\nLoading ontology:", base_owl_path)
    onto = get_ontology(f"file://{os.path.abspath(base_owl_path)}").load()

    ShrimpFarm = (
        onto.search_one(iri="*ShrimpFarm") or onto.search_one(label="ShrimpFarm")
    )
    if ShrimpFarm is None:
        raise ValueError("Could not find ShrimpFarm class in ontology.")

    from owlready2 import DataProperty

    def get_or_create_data_property(name_hint):
        prop = onto.search_one(iri=f"*{name_hint}") or onto.search_one(label=name_hint)
        if prop is None:
            with onto:
                class NewDP(DataProperty):
                    namespace = onto
            NewDP.name = name_hint
            prop = NewDP
            print(f"Created new data property: {name_hint}")
        return prop

    dp_temp = get_or_create_data_property("Temperature")
    dp_sal = get_or_create_data_property("Salinity")
    dp_stock = get_or_create_data_property("StockingDensity_PL_40MeterSquare")
    dp_prev = get_or_create_data_property("PreviousPrevalence")
    dp_risk = get_or_create_data_property("WSDRiskScore")
    dp_pred = get_or_create_data_property("PredictedWSD")

    print("Creating ShrimpFarm individuals and attaching predictions...")
    for idx_df, score, pred in zip(test_indices, hybrid_scores, hybrid_preds):
        row = df_all.iloc[idx_df]
        ind_name = f"Farm_{int(idx_df):04d}"
        farm_ind = ShrimpFarm(ind_name)

        farm_ind.__setattr__(dp_temp.python_name, [float(row[TEMPERATURE_COL])])
        farm_ind.__setattr__(dp_sal.python_name, [float(row[SALINITY_COL])])
        farm_ind.__setattr__(dp_stock.python_name, [float(row[STOCKING_COL])])
        farm_ind.__setattr__(dp_prev.python_name, [float(row[PREV_PREVALENCE_COL])])
        farm_ind.__setattr__(dp_risk.python_name, [float(score)])
        farm_ind.__setattr__(dp_pred.python_name, [int(pred)])

    print("Saving enriched ontology to:", output_owl_path)
    default_world.save(file=output_owl_path, format="rdfxml")


# ----------------------------------------------------------------------
# Main experiment (Algorithm 1)
# ----------------------------------------------------------------------

def run_experiment(
    data_path,
    ontology_path=None,
    output_owl_path=None,
    threshold=0.5,
    random_state=42,
):
    """
    Run Algorithm 1 on a given dataset.

    If ontology_path or output_owl_path is None, ontology writing is skipped.
    Returns metrics dict for LR, SVM, CNN, Hybrid.
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    df_all, X, y, feature_names = load_dataset(data_path)
    n_features = X.shape[1]

    # Stratified 70/15/15 split
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=0.15,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val,
        y_train_val,
        idx_train_val,
        test_size=0.1765,
        random_state=random_state,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_cnn = X_train_scaled.reshape(-1, n_features, 1)
    X_val_cnn = X_val_scaled.reshape(-1, n_features, 1)
    X_test_cnn = X_test_scaled.reshape(-1, n_features, 1)

    # LR
    lr = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        penalty="l2",
        random_state=random_state,
    )
    lr.fit(X_train_scaled, y_train)

    # SVM
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=random_state,
    )
    svm.fit(X_train_scaled, y_train)

    # CNN
    cnn = build_cnn(n_features)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )
    cnn.fit(
        X_train_cnn,
        y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Predictions
    p_lr_val = lr.predict_proba(X_val_scaled)[:, 1]
    p_svm_val = svm.predict_proba(X_val_scaled)[:, 1]
    p_cnn_val = cnn.predict(X_val_cnn).ravel()

    p_lr_test = lr.predict_proba(X_test_scaled)[:, 1]
    p_svm_test = svm.predict_proba(X_test_scaled)[:, 1]
    p_cnn_test = cnn.predict(X_test_cnn).ravel()

    df_val = df_all.iloc[idx_val].reset_index(drop=True)
    r_ont_val = np.array([ontology_risk_score(row) for _, row in df_val.iterrows()])

    df_test = df_all.iloc[idx_test].reset_index(drop=True)
    r_ont_test = np.array([ontology_risk_score(row) for _, row in df_test.iterrows()])

    # Weight search
    best_w = search_best_weights(
        y_val,
        p_lr_val,
        p_svm_val,
        p_cnn_val,
        r_ont_val,
        threshold=threshold,
    )
    w1, w2, w3, w4 = best_w

    # Hybrid
    R_hyb = (
        w1 * p_lr_test
        + w2 * p_svm_test
        + w3 * p_cnn_test
        + w4 * r_ont_test
    )
    y_pred_hyb = (R_hyb >= threshold).astype(int)

    # Metrics
    metrics_lr = evaluate_binary_classifier(
        y_test, p_lr_test, threshold, "Logistic Regression"
    )
    metrics_svm = evaluate_binary_classifier(
        y_test, p_svm_test, threshold, "SVM (RBF)"
    )
    metrics_cnn = evaluate_binary_classifier(
        y_test, p_cnn_test, threshold, "1D-CNN"
    )
    metrics_hyb = evaluate_binary_classifier(
        y_test, R_hyb, threshold, "Ontology-guided Ensemble (grid-searched)"
    )

    # Optional ontology writing
    if ontology_path is not None and output_owl_path is not None:
        attach_results_to_ontology(
            ontology_path, output_owl_path, df_all, idx_test, R_hyb, y_pred_hyb
        )

    return {
        "lr": metrics_lr,
        "svm": metrics_svm,
        "cnn": metrics_cnn,
        "hybrid": metrics_hyb,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ontology-guided ensemble learning for WSD (single run)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset (Dataset 1 or Dataset 2).",
    )
    parser.add_argument(
        "--ontology_path",
        type=str,
        default=None,
        help="Path to ShrimpDisease_Full.owl (optional).",
    )
    parser.add_argument(
        "--output_owl_path",
        type=str,
        default=None,
        help="Path to save enriched OWL (optional).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for splitting and model initialisation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_experiment(
        data_path=args.data_path,
        ontology_path=args.ontology_path,
        output_owl_path=args.output_owl_path,
        threshold=args.threshold,
        random_state=args.random_state,
    )

    print("\nSummary (AUC):")
    for name, m in results.items():
        print(f"{name:7s}: AUC = {m['auc']:.4f}")


if __name__ == "__main__":
    main()
