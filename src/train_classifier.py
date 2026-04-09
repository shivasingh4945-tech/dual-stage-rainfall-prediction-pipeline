"""
train_classifier.py
───────────────────
Trains XGBoost binary classifier for Rain / No Rain prediction.
Handles class imbalance with SMOTE (training set only).
Saves model, metrics, and feature importance.

Usage:
    python src/train_classifier.py
"""

import sys
import os
# Ensure src/ is on the path whether this script is run from the project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import joblib
import logging
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config_utils import load_config
from features import build_features_for_splits, get_feature_columns, fit_scaler, apply_scaler, prepare_Xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_splits(cfg):
    train = pd.read_csv(cfg["paths"]["train"], parse_dates=["date"])
    val   = pd.read_csv(cfg["paths"]["val"],   parse_dates=["date"])
    test  = pd.read_csv(cfg["paths"]["test"],  parse_dates=["date"])
    log.info(f"Loaded splits — Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def apply_smote(X_train, y_train, cfg):
    """
    SMOTE applied ONLY on training data.
    Oversamples minority class (Rain=1) to reduce imbalance.
    Never apply on val or test — would give unrealistically optimistic metrics.
    """
    k = cfg["imbalance"]["smote_k_neighbors"]
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    log.info(f"SMOTE applied: {len(X_train):,} → {len(X_res):,} rows")
    log.info(f"  Rain class after SMOTE: {y_res.mean()*100:.1f}%")
    return X_res, y_res


def build_classifier(cfg):
    params = cfg["xgb_classifier"].copy()
    # 'use_label_encoder' was removed in XGBoost 2.0 — strip defensively
    params.pop("use_label_encoder", None)
    # 'verbosity' must not be duplicated
    params.pop("verbosity", None)
    return XGBClassifier(**params, verbosity=0)


def evaluate_classifier(model, X, y, split_name=""):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "split":             split_name,
        "accuracy":          round(accuracy_score(y, y_pred), 4),
        "precision":         round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y, y_proba), 4),
        "avg_precision":     round(average_precision_score(y, y_proba), 4),
    }

    log.info(f"\n{'─'*50}")
    log.info(f"  {split_name} Results:")
    for k, v in metrics.items():
        if k != "split":
            log.info(f"    {k:20s}: {v}")
    log.info(f"\n{classification_report(y, y_pred, target_names=['No Rain','Rain'])}")
    return metrics


def run_training(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    os.makedirs(os.path.dirname(cfg["paths"]["model_clf"]), exist_ok=True)
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)

    # Load splits
    train, val, test = load_splits(cfg)

    # Feature engineering with carried-over history across split boundaries
    log.info("Building split-aware features...")
    train, val, test = build_features_for_splits(train=train, val=val, test=test)

    # Feature columns
    feat_cols = get_feature_columns(cfg)
    feat_cols = [c for c in feat_cols if c in train.columns]
    log.info(f"Using {len(feat_cols)} features")

    # Prepare X, y
    X_train, y_train = prepare_Xy(train, feat_cols, "rain_today")
    X_val,   y_val   = prepare_Xy(val,   feat_cols, "rain_today")
    X_test,  y_test  = prepare_Xy(test,  feat_cols, "rain_today")

    # Scale (fit on train only)
    scaler  = fit_scaler(X_train, cfg, save_path=cfg["paths"]["scaler"])
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    # SMOTE (train only)
    X_train_res, y_train_res = apply_smote(X_train, y_train, cfg)

    # Train
    log.info("Training XGBoost Classifier...")
    model = build_classifier(cfg)
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    train_metrics = evaluate_classifier(model, X_train, y_train, "Train")
    val_metrics   = evaluate_classifier(model, X_val,   y_val,   "Validation")
    test_metrics  = evaluate_classifier(model, X_test,  y_test,  "Test")

    # Save model
    joblib.dump(model, cfg["paths"]["model_clf"])
    log.info(f"Model saved → {cfg['paths']['model_clf']}")

    # Save metrics
    all_metrics = [train_metrics, val_metrics, test_metrics]
    metrics_path = os.path.join(cfg["paths"]["reports"], "classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    # Feature importance
    fi = pd.DataFrame({
        "feature":   feat_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(cfg["paths"]["reports"], "classifier_feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    log.info(f"Feature importance saved → {fi_path}")

    return model, scaler, all_metrics


if __name__ == "__main__":
    run_training()
