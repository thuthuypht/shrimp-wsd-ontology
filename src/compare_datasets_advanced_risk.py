#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ontology-guided ensemble learning for White Spot Disease (WSD) diagnosis
with a more informative, weighted ontology risk score.

Compared with wsd_ontology_ensemble.py, this version uses additional
conditions (Temperature, Salinity, pH, water depth, stocking density,
previous prevalence, crop rotation, reservoir / water source) and assigns
different weights to each rule. The weighted score is then mapped through
a sigmoid to approximate a probability r_ont in [0, 1].
"""

import argparse
import os
import math
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


# ----------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------

TARGET_COL = "VirusDetected"

TEMPERATURE_COL = "Temperature"
PH_COL = "pH"
SALINITY_COL = "Salinity"
STOCKING_COL = "StockingDensity_PL/40MeterSquare"
PREV_PREVALENCE_COL = "PreviousPrevalence(%)"
DEPTH_COL = "GherDepth_ft"
RESERVOIR_COL = "Reservoir"
CROPROT_COL = "CropRotation"
WATER_OTHER_FARMS_COL = "WaterComingViaOtherFarms"

# For ML models we still use the 4 core environmental features
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

    # Drop rows with missing values in main features + target
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
# Evaluation helper
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
# Weighted ontology-based risk score r_ont
# ----------------------------------------------------------------------

def ontology_risk_score_weighted(row):
    """
    More informative risk score r_ont in [0, 1] using multiple
    conditions and rule weights.

    Rules (example; adjust thresholds/weights according to domain knowledge):

    R1  (w=2.0): High temperature        -> Temperature > 32 °C
    R2  (w=1.5): Very low salinity       -> Salinity < 5 ppt
    R3  (w=2.5): Extreme pH              -> pH < 7.5 or pH > 8.5
    R4  (w=1.5): High stocking density   -> StockingDensity_PL/40m² > 80
    R5  (w=3.0): Previous WSD history    -> PreviousPrevalence(%) > 0
    R6  (w=1.0): Shallow water depth     -> GherDepth_ft < 3.0
    R7  (w=1.0): No crop rotation        -> CropRotation == 0
    R8  (w=1.5): Water from other farms with no reservoir
                                       -> WaterComingViaOtherFarms == 1
                                          and Reservoir == 0

    The final score is:
        r_norm = sum(w_i * I(rule_i)) / sum(w_i)
        r_ont  = sigmoid( gamma * (r_norm - 0.5) )
    where gamma controls steepness (here gamma = 5).
    """
    score = 0.0
    total_w = 0.0

    def add_rule(condition, weight):
        nonlocal score, total_w
        if condition:
            score += weight
        total_w += weight

    # Fetch values (comparisons with NaN give False automatically)
    T = row.get(TEMPERATURE_COL, np.nan)
    S = row.get(SALINITY_COL, np.nan)
    pH = row.get(PH_COL, np.nan)
    stock = row.get(STOCKING_COL, np.nan)
    prev = row.get(PREV_PREVALENCE_COL, np.nan)
    depth = row.get(DEPTH_COL, np.nan)
    crop_rot = row.get(CROPROT_COL, np.nan)
    reservoir = row.get(RESERVOIR_COL, np.nan)
    water_other = row.get(WATER_OTHER_FARMS_COL, np.nan)

    # R1: High temperature
    add_rule(T > 32, 2.0)

    # R2: Very low salinity
    add_rule(S < 5, 1.5)

    # R3: Extreme pH
    add_rule((pH < 7.5) or (pH > 8.5), 2.5)

    # R4: High stocking density (upper tail of observed distribution)
    add_rule(stock > 80, 1.5)

    # R5: Previous WSD history
    add_rule(prev > 0, 3.0)

    # R6: Shallow water depth
    add_rule(depth < 3.0, 1.0)

    # R7: No crop rotation
    add_rule(crop_rot == 0, 1.0)

    # R8: Water comes via other farms and no reservoir
    add_rule((water_other == 1) and (reservoir == 0), 1.5)

    if total_w == 0:
        return 0.5  # neutral if no information

    r_norm = score / total_w  # [0, 1]
    gamma = 5.0
    r_sigmoid = 1.0 / (1.0 + math.exp(-gamma * (r_norm - 0.5)))
    return r_sigmoid


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
    print("  w1 = LR,  w2 = SVM,  w3 = CNN,  w4 = ontology-risk (weighted)")
    return best_w


# ----------------------------------------------------------------------
# Ontology writing (optional – same as before)
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
    dp_risk = get_or_create_data_property("WSDRiskScoreWeighted")
    dp_pred = get_or_create_data_property("PredictedWSDWeighted")

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
# Main experiment (Algorithm 1, weighted r_ont)
# ----------------------------------------------------------------------

def run_experiment(
    data_path,
    ontology_path=None,
    output_owl_path=None,
    threshold=0.5,
    random_state=42,
):
    """
    Run Algorithm 1 on a given dataset using the weighted ontology risk.

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

    # Ontology risk on validation & test
    df_val = df_all.iloc[idx_val].reset_index(drop=True)
    r_ont_val = np.array(
        [ontology_risk_score_weighted(row) for _, row in df_val.iterrows()]
    )

    df_test = df_all.iloc[idx_test].reset_index(drop=True)
    r_ont_test = np.array(
        [ontology_risk_score_weighted(row) for _, row in df_test.iterrows()]
    )

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
        y_test, R_hyb, threshold, "Ontology-guided Ensemble (weighted r_ont)"
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
        description="Ontology-guided ensemble learning for WSD (weighted ontology risk)."
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
