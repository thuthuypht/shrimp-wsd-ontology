#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ontology-guided ensemble learning for White Spot Disease (WSD) diagnosis.

Implements "Algorithm 1 for Ontology-guided Ensemble Learning for WSD Diagnosis"
on the dataset WSSVRiskFactor_rev.xlsx and the ontology ShrimpDisease_Full.owl.

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


# ----------------------------------------------------------------------
# Configuration – matches WSSVRiskFactor_rev.xlsx
# ----------------------------------------------------------------------

# Target column: 1 = WSD positive, 0 = WSD negative
TARGET_COL = "VirusDetected"

# Feature columns used by ML models and ontology-based scoring
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

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Data loading and basic preprocessing
# ----------------------------------------------------------------------

def load_dataset(path: str):
    """
    Load dataset from Excel/CSV and return:
        df (DataFrame), X (features), y (labels), feature_names.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in file {path}: {missing}")

    # Simple strategy: drop rows with missing values in key columns
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

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
# Ontology-based risk score r_ont
# ----------------------------------------------------------------------

def ontology_risk_score(row):
    """
    Example ontology-inspired risk score r_ont in [0, 1].

    You can align these thresholds with your SWRL rules.
    Here we use a simple scheme:

    - Temperature > 32°C              -> +1
    - Salinity < 5 ppt                -> +1
    - Stocking density > 25 PL/40m²   -> +1
    - PreviousPrevalence(%) > 0       -> +1

    r_ont = (number of triggered conditions) / 4
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
# Ontology writing
# ----------------------------------------------------------------------

def attach_results_to_ontology(
    base_owl_path: str,
    output_owl_path: str,
    df_all: pd.DataFrame,
    test_indices: np.ndarray,
    hybrid_scores: np.ndarray,
    hybrid_preds: np.ndarray,
):
    """
    Load the base ShrimpDisease_Full.owl, create a ShrimpFarm individual
    for each test farm, attach features + predictions, and save to a new OWL.

    NOTE: Class and property names are taken from ShrimpDisease_Full.owl.
    If you have changed them, adapt this function.
    """
    print("\nLoading ontology:", base_owl_path)
    onto = get_ontology(f"file://{os.path.abspath(base_owl_path)}").load()

    # Find ShrimpFarm class
    ShrimpFarm = (
        onto.search_one(iri="*ShrimpFarm") or onto.search_one(label="ShrimpFarm")
    )
    if ShrimpFarm is None:
        raise ValueError("Could not find ShrimpFarm class in ontology.")

    # Convenience: create or retrieve data properties by name hint
    def get_or_create_data_property(name_hint):
        from owlready2 import DataProperty

        prop = onto.search_one(iri=f"*{name_hint}") or onto.search_one(label=name_hint)
        if prop is None:
            # Create a new data property if not present
            with onto:
                class NewDP(DataProperty):
                    namespace = onto
            NewDP.name = name_hint
            prop = NewDP
            print(f"Created new data property: {name_hint}")
        return prop

    # Ontology properties for input features
    dp_temp = get_or_create_data_property("Temperature")
    dp_sal = get_or_create_data_property("Salinity")
    dp_stock = get_or_create_data_property("StockingDensity_PL_40MeterSquare")
    dp_prev = get_or_create_data_property("PreviousPrevalence")

    # Properties for our predictions
    dp_risk = get_or_create_data_property("WSDRiskScore")
    dp_pred = get_or_create_data_property("PredictedWSD")

    print("Creating ShrimpFarm individuals and attaching predictions...")
    for idx_df, score, pred in zip(test_indices, hybrid_scores, hybrid_preds):
        row = df_all.iloc[idx_df]

        # Individual name: Farm_0001, Farm_0002, ...
        ind_name = f"Farm_{int(idx_df):04d}"
        farm_ind = ShrimpFarm(ind_name)

        # Attach raw features (if property exists)
        farm_ind.__setattr__(dp_temp.python_name, [float(row[TEMPERATURE_COL])])
        farm_ind.__setattr__(dp_sal.python_name, [float(row[SALINITY_COL])])
        farm_ind.__setattr__(dp_stock.python_name, [float(row[STOCKING_COL])])
        farm_ind.__setattr__(dp_prev.python_name, [float(row[PREV_PREVALENCE_COL])])

        # Attach hybrid risk score and predicted label
        farm_ind.__setattr__(dp_risk.python_name, [float(score)])
        farm_ind.__setattr__(dp_pred.python_name, [int(pred)])

    print("Saving enriched ontology to:", output_owl_path)
    default_world.save(file=output_owl_path, format="rdfxml")


# ----------------------------------------------------------------------
# Main experiment (Algorithm 1)
# ----------------------------------------------------------------------

def run_experiment(
    data_path,
    ontology_path,
    output_owl_path,
    w1=0.25,
    w2=0.25,
    w3=0.25,
    w4=0.25,
    threshold=0.5,
):
    """
    Execute Algorithm 1:
    - load WSSVRiskFactor_rev.xlsx
    - split into train / val / test
    - train LR, SVM, CNN
    - compute ontology risk scores
    - combine into hybrid ensemble
    - evaluate & save ontology with results
    """

    # Reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # 1. Load dataset
    df_all, X, y, feature_names = load_dataset(data_path)
    n_features = X.shape[1]

    # 2. Split 70 / 15 / 15 (stratified)
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val,
        y_train_val,
        idx_train_val,
        test_size=0.1765,  # 0.1765 * 0.85 ≈ 0.15 of whole dataset
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    # 3. Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # CNN expects (samples, timesteps, channels)
    X_train_cnn = X_train_scaled.reshape(-1, n_features, 1)
    X_val_cnn = X_val_scaled.reshape(-1, n_features, 1)
    X_test_cnn = X_test_scaled.reshape(-1, n_features, 1)

    # 4. Train base ML models
    # 4.1 Logistic Regression
    lr = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        penalty="l2",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train_scaled, y_train)

    # 4.2 SVM (RBF kernel) with probability calibration
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
    )
    svm.fit(X_train_scaled, y_train)

    # 4.3 1D-CNN
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

    # 5. Probabilistic predictions on test set
    p_lr_test = lr.predict_proba(X_test_scaled)[:, 1]
    p_svm_test = svm.predict_proba(X_test_scaled)[:, 1]
    p_cnn_test = cnn.predict(X_test_cnn).ravel()

    # 6. Ontology-based risk scores r_ont on test set
    df_test = df_all.iloc[idx_test].reset_index(drop=True)
    r_ont_test = np.array(
        [ontology_risk_score(row) for _, row in df_test.iterrows()]
    )

    # 7. Hybrid ensemble: R_hyb = w1 * p_LR + w2 * p_SVM + w3 * p_CNN + w4 * r_ont
    w_sum = w1 + w2 + w3 + w4
    w1, w2, w3, w4 = w1 / w_sum, w2 / w_sum, w3 / w_sum, w4 / w_sum

    R_hyb = (
        w1 * p_lr_test
        + w2 * p_svm_test
        + w3 * p_cnn_test
        + w4 * r_ont_test
    )
    y_pred_hyb = (R_hyb >= threshold).astype(int)

    # 8. Evaluate and compare models
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
        y_test, R_hyb, threshold, "Ontology-guided Ensemble (proposed)"
    )

    # 9. Save enriched OWL with predictions
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
# Command-line interface
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ontology-guided ensemble learning for WSD diagnosis."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to WSSVRiskFactor_rev.xlsx (or CSV).",
    )
    parser.add_argument(
        "--ontology_path",
        type=str,
        required=True,
        help="Path to base ShrimpDisease_Full.owl.",
    )
    parser.add_argument(
        "--output_owl_path",
        type=str,
        default="ShrimpDisease_Full_results.owl",
        help="Path to save enriched OWL file with predictions.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        metavar=("w1", "w2", "w3", "w4"),
        default=[0.25, 0.25, 0.25, 0.25],
        help="Ensemble weights for LR, SVM, CNN, ontology risk respectively.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying WSD-positive farms.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    w1, w2, w3, w4 = args.weights

    results = run_experiment(
        data_path=args.data_path,
        ontology_path=args.ontology_path,
        output_owl_path=args.output_owl_path,
        w1=w1,
        w2=w2,
        w3=w3,
        w4=w4,
        threshold=args.threshold,
    )

    print("\nSummary (AUC):")
    for name, m in results.items():
        print(f"{name:7s}: AUC = {m['auc']:.4f}")


if __name__ == "__main__":
    main()
