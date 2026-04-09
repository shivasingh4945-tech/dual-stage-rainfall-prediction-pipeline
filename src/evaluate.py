"""
evaluate.py
───────────
Comprehensive model evaluation — loads saved models, runs
evaluation on test set, prints full metrics report, and
saves a consolidated summary.

Usage:
    python src/evaluate.py
"""

import sys
import os
# Ensure src/ is on the path whether this script is run from the project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import json
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from config_utils import load_config
from features import build_features_for_splits, get_feature_columns, apply_scaler, prepare_Xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def print_banner(title):
    log.info("\n" + "═"*60)
    log.info(f"  {title}")
    log.info("═"*60)


def evaluate_all(config_path="configs/config.yaml"):
    cfg = load_config(config_path)

    # ── Load data ──────────────────────────────────────────────
    train = pd.read_csv(cfg["paths"]["train"], parse_dates=["date"])
    val   = pd.read_csv(cfg["paths"]["val"],   parse_dates=["date"])
    test  = pd.read_csv(cfg["paths"]["test"],  parse_dates=["date"])
    _, _, test = build_features_for_splits(train=train, val=val, test=test)

    feat_cols = get_feature_columns(cfg)
    feat_cols = [c for c in feat_cols if c in test.columns]

    # ── Classification ─────────────────────────────────────────
    print_banner("CLASSIFICATION EVALUATION — Rain / No Rain")

    clf   = joblib.load(cfg["paths"]["model_clf"])
    scaler_clf = joblib.load(cfg["paths"]["scaler"])

    X_test, y_test = prepare_Xy(test, feat_cols, "rain_today")
    X_test_sc = apply_scaler(X_test, scaler_clf)

    y_pred  = clf.predict(X_test_sc)
    y_proba = clf.predict_proba(X_test_sc)[:, 1]

    log.info("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    log.info(f"\n  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    log.info(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    log.info(f"\n  Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
    log.info(f"  F1 Score       : {f1_score(y_test, y_pred):.4f}")
    log.info(f"  ROC-AUC        : {roc_auc_score(y_test, y_proba):.4f}")
    log.info(f"  Avg Precision  : {average_precision_score(y_test, y_proba):.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['No Rain','Rain'])}")

    # ── Regression ─────────────────────────────────────────────
    print_banner("REGRESSION EVALUATION — Rainfall Amount (mm/day)")

    reg    = joblib.load(cfg["paths"]["model_reg"])
    scaler_reg = joblib.load(cfg["paths"]["scaler"].replace(".pkl", "_reg.pkl"))

    X_test_r, y_test_r = prepare_Xy(test, feat_cols, "PRECTOTCORR", log_transform=True)
    X_test_r_sc = apply_scaler(X_test_r, scaler_reg)

    y_pred_log = reg.predict(X_test_r_sc)
    y_true_mm  = np.expm1(y_test_r)
    y_pred_mm  = np.expm1(y_pred_log).clip(min=0)

    rmse = np.sqrt(mean_squared_error(y_true_mm, y_pred_mm))
    mae  = mean_absolute_error(y_true_mm, y_pred_mm)
    r2   = r2_score(y_true_mm, y_pred_mm)

    log.info(f"\n  All Days:")
    log.info(f"    RMSE : {rmse:.4f} mm")
    log.info(f"    MAE  : {mae:.4f} mm")
    log.info(f"    R²   : {r2:.4f}")

    mask_rain = y_true_mm > 2.5
    if mask_rain.sum() > 0:
        rmse_r = np.sqrt(mean_squared_error(y_true_mm[mask_rain], y_pred_mm[mask_rain]))
        mae_r  = mean_absolute_error(y_true_mm[mask_rain], y_pred_mm[mask_rain])
        r2_r   = r2_score(y_true_mm[mask_rain], y_pred_mm[mask_rain])
        log.info(f"\n  Rainy Days Only (>{2.5}mm) — {mask_rain.sum():,} days:")
        log.info(f"    RMSE : {rmse_r:.4f} mm")
        log.info(f"    MAE  : {mae_r:.4f} mm")
        log.info(f"    R²   : {r2_r:.4f}")

    # ── Summary Report ──────────────────────────────────────────
    print_banner("SUMMARY REPORT")
    summary = {
        "classification": {
            "accuracy":       round(accuracy_score(y_test, y_pred), 4),
            "f1":             round(f1_score(y_test, y_pred), 4),
            "roc_auc":        round(roc_auc_score(y_test, y_proba), 4),
            "avg_precision":  round(average_precision_score(y_test, y_proba), 4),
        },
        "regression": {
            "rmse_all":  round(rmse, 4),
            "mae_all":   round(mae, 4),
            "r2_all":    round(r2, 4),
        }
    }
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    summary_path = os.path.join(cfg["paths"]["reports"], "final_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary saved → {summary_path}")
    return summary


if __name__ == "__main__":
    evaluate_all()
