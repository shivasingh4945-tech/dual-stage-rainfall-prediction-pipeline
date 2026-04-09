"""
train_regressor.py
──────────────────
Trains XGBoost + LightGBM regressors for rainfall amount prediction.
Uses NASA PRECTOTCORR as target (log1p transformed).
Evaluates with RMSE, MAE, R², MAPE on rainy days only.

Usage:
    python src/train_regressor.py
"""

import sys
import os
# Ensure src/ is on the path whether this script is run from the project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import logging
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config_utils import load_config
from features import build_features_for_splits, get_feature_columns, fit_scaler, apply_scaler, prepare_Xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_splits(cfg):
    train = pd.read_csv(cfg["paths"]["train"], parse_dates=["date"])
    val   = pd.read_csv(cfg["paths"]["val"],   parse_dates=["date"])
    test  = pd.read_csv(cfg["paths"]["test"],  parse_dates=["date"])
    return train, val, test


def mape(y_true, y_pred, eps=1.0):
    """MAPE only on rainy days (y_true >= eps) to avoid div-by-zero."""
    mask = y_true >= eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_regressor(model, X, y_log, split_name="", log_transform=True):
    """
    Evaluate in original space (mm/day) after inverse log transform.
    Metrics computed on:
      1. All days
      2. Rainy days only (>2.5mm) — more meaningful for heavy rain events
    """
    y_pred_log = model.predict(X)

    if log_transform:
        y_true = np.expm1(y_log)
        y_pred = np.expm1(y_pred_log).clip(min=0)
    else:
        y_true = y_log
        y_pred = y_pred_log.clip(min=0)

    rmse_all  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_all   = mean_absolute_error(y_true, y_pred)
    r2_all    = r2_score(y_true, y_pred)
    mape_all  = mape(y_true.values, y_pred)

    # Rainy days only
    mask_rain = y_true > 2.5
    if mask_rain.sum() > 0:
        rmse_rain = np.sqrt(mean_squared_error(y_true[mask_rain], y_pred[mask_rain]))
        mae_rain  = mean_absolute_error(y_true[mask_rain], y_pred[mask_rain])
        r2_rain   = r2_score(y_true[mask_rain], y_pred[mask_rain])
        mape_rain = mape(y_true[mask_rain].values, y_pred[mask_rain])
    else:
        rmse_rain = mae_rain = r2_rain = mape_rain = np.nan

    def _safe_round(v, n):
        """round() that converts NaN → None (JSON null) to stay serializable."""
        import math
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(v, n)

    metrics = {
        "split":       split_name,
        "rmse_all":    _safe_round(rmse_all, 4),
        "mae_all":     _safe_round(mae_all, 4),
        "r2_all":      _safe_round(r2_all, 4),
        "mape_all":    _safe_round(mape_all, 2),
        "rmse_rain":   _safe_round(rmse_rain, 4),
        "mae_rain":    _safe_round(mae_rain, 4),
        "r2_rain":     _safe_round(r2_rain, 4),
        "mape_rain":   _safe_round(mape_rain, 2),
        "n_rainy_days": int(mask_rain.sum()),
    }

    log.info(f"\n{'─'*50}")
    log.info(f"  {split_name} Regressor Results:")
    log.info(f"    All days  — RMSE: {rmse_all:.3f}  MAE: {mae_all:.3f}  R²: {r2_all:.4f}  MAPE: {mape_all:.1f}%")
    rain_mape_str = f"{mape_rain:.1f}%" if mape_rain is not None and not (isinstance(mape_rain, float) and __import__('math').isnan(mape_rain)) else "N/A"
    log.info(f"    Rain days — RMSE: {rmse_rain:.3f}  MAE: {mae_rain:.3f}  R²: {r2_rain:.4f}  MAPE: {rain_mape_str}")
    return metrics, y_true, y_pred


def run_training(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    os.makedirs(os.path.dirname(cfg["paths"]["model_reg"]), exist_ok=True)
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["predictions"]) or ".", exist_ok=True)

    train, val, test = load_splits(cfg)

    log.info("Building split-aware features...")
    train, val, test = build_features_for_splits(train=train, val=val, test=test)

    feat_cols = get_feature_columns(cfg)
    feat_cols = [c for c in feat_cols if c in train.columns]
    log.info(f"Using {len(feat_cols)} features")

    log_transform = cfg["target"]["log_transform"]

    X_train, y_train = prepare_Xy(train, feat_cols, "PRECTOTCORR", log_transform=log_transform)
    X_val,   y_val   = prepare_Xy(val,   feat_cols, "PRECTOTCORR", log_transform=log_transform)
    X_test,  y_test  = prepare_Xy(test,  feat_cols, "PRECTOTCORR", log_transform=log_transform)

    # Scale
    scaler_path = cfg["paths"]["scaler"].replace(".pkl", "_reg.pkl")
    scaler  = fit_scaler(X_train, cfg, save_path=scaler_path)
    X_train = apply_scaler(X_train, scaler)
    X_val   = apply_scaler(X_val,   scaler)
    X_test  = apply_scaler(X_test,  scaler)

    all_metrics = {}

    # ── XGBoost Regressor ─────────────────────────────────────────────────────
    log.info("\nTraining XGBoost Regressor...")
    xgb_params = {k: v for k, v in cfg["xgb_regressor"].items()}
    xgb_params.pop("verbosity", None)
    xgb_reg = XGBRegressor(**xgb_params, verbosity=0)
    xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    xgb_train_m, _, _  = evaluate_regressor(xgb_reg, X_train, y_train, "XGB Train",  log_transform)
    xgb_val_m,   _, _  = evaluate_regressor(xgb_reg, X_val,   y_val,   "XGB Val",    log_transform)
    xgb_test_m, y_true, xgb_pred = evaluate_regressor(xgb_reg, X_test, y_test, "XGB Test", log_transform)

    joblib.dump(xgb_reg, cfg["paths"]["model_reg"])
    log.info(f"XGBoost Regressor saved → {cfg['paths']['model_reg']}")
    all_metrics["xgb"] = [xgb_train_m, xgb_val_m, xgb_test_m]

    # ── LightGBM Regressor ────────────────────────────────────────────────────
    log.info("\nTraining LightGBM Regressor...")
    lgbm_params = {k: v for k, v in cfg["lgbm_regressor"].items()}
    lgbm_reg = LGBMRegressor(**lgbm_params, verbosity=-1)
    lgbm_reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    lgbm_train_m, _, _     = evaluate_regressor(lgbm_reg, X_train, y_train, "LGBM Train", log_transform)
    lgbm_val_m,   _, _     = evaluate_regressor(lgbm_reg, X_val,   y_val,   "LGBM Val",   log_transform)
    lgbm_test_m, _, lgbm_pred = evaluate_regressor(lgbm_reg, X_test, y_test, "LGBM Test", log_transform)

    joblib.dump(lgbm_reg, cfg["paths"]["model_reg"].replace("xgb", "lgbm"))
    all_metrics["lgbm"] = [lgbm_train_m, lgbm_val_m, lgbm_test_m]

    # ── Ensemble (simple average) ──────────────────────────────────────────────
    log.info("\nEnsemble (XGB + LGBM average)...")
    ensemble_pred = (xgb_pred + lgbm_pred) / 2
    mask_rain = y_true > 2.5
    ens_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
    ens_r2   = r2_score(y_true, ensemble_pred)
    log.info(f"  Ensemble Test — RMSE: {ens_rmse:.3f}  R²: {ens_r2:.4f}")

    # Save predictions
    pred_df = test[["district", "date"]].copy()
    pred_df = pred_df.iloc[:len(y_true)].copy()
    pred_df["actual_mm"]   = y_true.values
    pred_df["xgb_pred_mm"] = xgb_pred
    pred_df["lgbm_pred_mm"] = lgbm_pred
    pred_df["ensemble_mm"] = ensemble_pred
    pred_df.to_csv(cfg["paths"]["predictions"], index=False)
    log.info(f"Predictions saved → {cfg['paths']['predictions']}")

    # Save metrics
    metrics_path = os.path.join(cfg["paths"]["reports"], "regressor_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    return xgb_reg, lgbm_reg, all_metrics


if __name__ == "__main__":
    run_training()
