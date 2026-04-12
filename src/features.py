"""
features.py
───────────
All feature engineering — lag features, rolling stats,
wind encoding, temporal features, scaling, encoding.

Usage:
    from src.features import build_features, scale_features
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)


# ── Temporal Features ─────────────────────────────────────────────────────────
def add_temporal_features(df):
    """
    Month sin/cos : circular encoding so Jan and Dec are adjacent
    Season        : 4 Indian meteorological seasons
    Day of year   : captures fine-grained seasonality
    """
    df = df.copy()
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"]        = df["date"].dt.year

    # Circular month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Season (Indian meteorological calendar)
    season_map = {
        1: "Winter",       2: "Winter",
        3: "Pre_Monsoon",  4: "Pre_Monsoon",  5: "Pre_Monsoon",
        6: "Monsoon",      7: "Monsoon",      8: "Monsoon",      9: "Monsoon",
        10: "Post_Monsoon", 11: "Post_Monsoon", 12: "Post_Monsoon"
    }
    df["season"] = df["month"].map(season_map)

    # Ordinal encode season for models
    season_order = {"Winter": 0, "Pre_Monsoon": 1, "Monsoon": 2, "Post_Monsoon": 3}
    df["season_encoded"] = df["season"].map(season_order)

    log.info("Temporal features added (month_sin/cos, season, day_of_year)")
    return df


# ── Wind Features ─────────────────────────────────────────────────────────────
def add_wind_features(df):
    """
    Wind speed from ERA5 U/V components : sqrt(u10^2 + v10^2)
    Wind direction from U/V             : encoded as sin/cos (circular)
    NASA wind directions WD10M, WD50M   : encoded as sin/cos
    """
    df = df.copy()

    # ERA5 wind speed and direction from components
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed_10m"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        # Meteorological convention: direction wind is blowing FROM
        wind_dir_rad = np.arctan2(-df["u10"], -df["v10"])
        df["wind_dir_sin"] = np.sin(wind_dir_rad)
        df["wind_dir_cos"] = np.cos(wind_dir_rad)

    # NASA wind direction encoding (circular — 0° and 360° are same direction)
    for col in ["WD10M", "WD50M"]:
        if col in df.columns:
            rad = np.radians(df[col])
            df[f"{col.lower()}_sin"] = np.sin(rad)
            df[f"{col.lower()}_cos"] = np.cos(rad)

    log.info("Wind features added (wind_speed_10m, wind_dir_sin/cos, wd10m/50m_sin/cos)")
    return df


# ── Thermodynamic Interaction Features ───────────────────────────────────────
def add_thermo_features(df):
    """
    TDD (Temperature-Dewpoint Depression) : T2m - D2m
        When TDD < 3°C → atmosphere near saturation → rainfall likely
    Humidity stress index : RH2M / (TDD + 0.1)
        High RH + low TDD → extreme moisture signal
    """
    df = df.copy()

    if "t2m" in df.columns and "d2m" in df.columns:
        df["tdd"] = df["t2m"] - df["d2m"]     # Temperature-Dewpoint Depression

    if "RH2M" in df.columns and "tdd" in df.columns:
        df["humidity_stress"] = df["RH2M"] / (df["tdd"].clip(lower=0.1))

    log.info("Thermodynamic features added (tdd, humidity_stress)")
    return df


# ── Lag and Rolling Features ──────────────────────────────────────────────────
def add_lag_features(df):
    """
    Lag features capture rainfall persistence (autocorrelation r≈0.55 at lag-1).
    Rolling means capture active/break monsoon cycles and soil moisture memory.

    CRITICAL: All lags and rolling stats are computed PER DISTRICT.
    Computing globally would mix time series from different districts.
    """
    df = df.copy()
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    # Precipitation lags (using NASA PRECTOTCORR — the reliable source)
    for lag in [1, 3, 7]:
        df[f"precip_lag_{lag}d"] = df.groupby("district")["PRECTOTCORR"].shift(lag)

    # Rolling means (per district)
    for window in [7, 30]:
        df[f"precip_roll_{window}d"] = (
            df.groupby("district")["PRECTOTCORR"]
            .transform(lambda x: x.rolling(window, min_periods=max(3, window // 3)).mean())
        )

    # Prior rain streak: consecutive rainy days observed before the current day.
    # Shifting keeps this feature causal and avoids leaking today's target.
    if "rain_today" in df.columns:
        current_streak = (
            df.groupby("district")["rain_today"]
            .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
            * df["rain_today"]
        )
        df["rain_streak"] = (
            current_streak.groupby(df["district"]).shift(1).fillna(0).astype(int)
        )
    else:
        df["rain_streak"] = 0

    n_lag_nulls = df[["precip_lag_1d","precip_lag_7d","precip_roll_7d"]].isnull().sum().sum()
    log.info(f"Lag & rolling features added — {n_lag_nulls:,} NaNs at series start (will be dropped)")
    return df


# ── Build Full Feature Set ────────────────────────────────────────────────────
def build_features(df):
    """
    Master function — applies all feature engineering steps in order.
    Call this on train, val, and test separately (after fitting scaler on train).
    """
    log.info("Building features...")
    df = add_temporal_features(df)
    df = add_wind_features(df)
    df = add_thermo_features(df)
    df = add_lag_features(df)
    log.info(f"  Total columns after feature engineering: {df.shape[1]}")
    return df


def build_features_for_splits(train=None, val=None, test=None):
    """
    Build features on a concatenated chronological frame so validation and test
    rows can use historical context from earlier splits.
    """
    split_frames = [
        (name, frame.copy())
        for name, frame in [("train", train), ("val", val), ("test", test)]
        if frame is not None
    ]
    if not split_frames:
        return tuple()

    tag_col = "__split__"
    combined = pd.concat(
        [frame.assign(**{tag_col: name}) for name, frame in split_frames],
        ignore_index=True,
    )
    combined = build_features(combined)

    outputs = []
    for name, _ in split_frames:
        split_df = combined.loc[combined[tag_col] == name].drop(columns=[tag_col])
        split_df = drop_lag_nulls(split_df)
        outputs.append(split_df)
        log.info(
            "  %s feature set ready: %s rows (%s → %s)",
            name,
            f"{len(split_df):,}",
            split_df["date"].min().date(),
            split_df["date"].max().date(),
        )
    return tuple(outputs)


# ── Get Final Feature Columns ─────────────────────────────────────────────────
def get_feature_columns(cfg):
    """
    Returns the final list of feature columns used for model training.
    Reads from config — single source of truth.
    """
    drop_cols = set(cfg["features"]["drop"] + ["rain_today", "season", "month", "year"])
    # All engineered + base features
    all_features = (
        cfg["features"]["era5"] +
        cfg["features"]["nasa"] +
        cfg["features"]["spatial"] +
        cfg["features"]["engineered"]
    )
    return [f for f in all_features if f not in drop_cols]


# ── Drop Lag NaNs ─────────────────────────────────────────────────────────────
def drop_lag_nulls(df):
    """
    Lag features produce NaN for the first N rows of each district's time series.
    These MUST be dropped — imputing lag NaNs would contaminate the temporal signal.
    """
    lag_cols = [c for c in df.columns if "lag" in c or "roll" in c]
    before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    log.info(f"  Dropped {before - len(df):,} rows with lag NaNs → {len(df):,} rows remain")
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────
def fit_scaler(X_train, cfg, save_path=None):
    """
    Fit RobustScaler on training data ONLY.
    RobustScaler uses median and IQR → immune to extreme precipitation outliers.
    NEVER fit on val or test — that would be data leakage.
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        log.info(f"  Scaler saved → {save_path}")
    return scaler


def apply_scaler(X, scaler):
    """Transform features using a pre-fitted scaler."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


# ── Prepare X, y ──────────────────────────────────────────────────────────────
def prepare_Xy(df, feature_cols, target_col, log_transform=False):
    """
    Extract feature matrix X and target vector y from a dataframe.
    Optionally log1p-transform the regression target.
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning(f"  Missing feature columns: {missing}")

    X = df[available].copy()
    y = df[target_col].copy()

    if log_transform:
        y = np.log1p(y)
        log.info(f"  Applied log1p transform to target '{target_col}'")

    return X, y
