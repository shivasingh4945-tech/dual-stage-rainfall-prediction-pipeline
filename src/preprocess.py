"""
preprocess.py
─────────────
Loads the three raw datasets, applies all cleaning steps,
merges them, and saves the processed output to data/processed/.

Usage:
    python src/preprocess.py
"""

import pandas as pd
import os
import logging
from config_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Load Config ──────────────────────────────────────────────────────────────
# ── Step 1: Load Raw Data ────────────────────────────────────────────────────
def load_raw(cfg):
    log.info("Loading raw datasets...")
    era5   = pd.read_csv(cfg["paths"]["raw_era5"],   parse_dates=["date"])
    nasa   = pd.read_csv(cfg["paths"]["raw_nasa"],   parse_dates=["date"])
    coords = pd.read_csv(cfg["paths"]["raw_coords"])
    log.info(f"  ERA5:   {era5.shape}")
    log.info(f"  NASA:   {nasa.shape}")
    log.info(f"  Coords: {coords.shape}")
    return era5, nasa, coords


# ── Step 2: Normalize District Names ─────────────────────────────────────────
def normalize_districts(era5, nasa, coords):
    log.info("Normalizing district names...")
    era5["district"]   = era5["district"].str.lower().str.strip().str.replace(" ", "_")
    nasa["district"]   = nasa["district"].str.lower().str.strip().str.replace(" ", "_")
    coords["district"] = coords["district"].str.lower().str.strip().str.replace(" ", "_")
    return era5, nasa, coords


# ── Step 3: Drop kanpur Duplicate from NASA ───────────────────────────────────
def drop_kanpur_duplicate(nasa, cfg):
    if cfg["preprocessing"]["drop_kanpur_duplicate"]:
        before = len(nasa)
        nasa = nasa[nasa["district"] != "kanpur"].copy()
        log.info(f"Dropped 'kanpur' duplicate from NASA: {before - len(nasa)} rows removed")
    return nasa


# ── Step 4: ERA5 Unit Conversions ─────────────────────────────────────────────
def convert_era5_units(era5):
    """
    ERA5 raw units → human-readable units.
    t2m, d2m : Kelvin       → °C    (subtract 273.15)
    sp       : Pascals      → hPa   (divide by 100)
    tp       : metres/day   → mm/day (multiply by 1000)
    u10, v10 : m/s          → no change
    """
    log.info("Converting ERA5 units...")
    era5 = era5.copy()
    era5["t2m"] = era5["t2m"] - 273.15
    era5["d2m"] = era5["d2m"] - 273.15
    era5["sp"]  = era5["sp"]  / 100.0
    era5["tp"]  = era5["tp"]  * 1000.0
    log.info(f"  t2m range: {era5['t2m'].min():.1f}°C → {era5['t2m'].max():.1f}°C")
    log.info(f"  sp  range: {era5['sp'].min():.1f} → {era5['sp'].max():.1f} hPa")
    log.info(f"  tp  range: {era5['tp'].min():.3f} → {era5['tp'].max():.2f} mm/day")
    return era5


# ── Step 5: Winsorize NASA Precipitation Outliers ────────────────────────────
def winsorize_nasa_precip(nasa, cfg):
    """
    Cap PRECTOTCORR at the 99th percentile.
    Reason: 279mm on a February day in Hardoi is a satellite artifact.
    83 values exceed 100mm — physically impossible for winter UP.
    """
    pct = cfg["preprocessing"]["nasa_precip_winsorize_pct"]
    cap = nasa["PRECTOTCORR"].quantile(pct / 100)
    before_max = nasa["PRECTOTCORR"].max()
    nasa = nasa.copy()
    nasa["PRECTOTCORR"] = nasa["PRECTOTCORR"].clip(upper=cap)
    log.info(f"Winsorized PRECTOTCORR at {pct}th pct: {before_max:.2f}mm → capped at {cap:.2f}mm")
    return nasa


# ── Step 6: Merge All Three Datasets ─────────────────────────────────────────
def merge_datasets(era5, nasa, coords):
    """
    ERA5 × NASA : INNER JOIN on [district, date]
    Result × Coords : LEFT JOIN on [district]
    """
    log.info("Merging ERA5 + NASA POWER...")
    merged = pd.merge(era5, nasa, on=["district", "date"], how="inner")
    log.info(f"  After ERA5 × NASA merge: {merged.shape}")

    log.info("Attaching coordinates...")
    merged = pd.merge(merged, coords, on="district", how="left")
    log.info(f"  After attaching coords: {merged.shape}")

    missing_lat = merged["latitude"].isna().sum()
    if missing_lat > 0:
        log.warning(f"  {missing_lat} rows missing latitude after coord join!")
    else:
        log.info("  All districts matched coordinates ✅")

    return merged


# ── Step 7: Data Quality Checks ───────────────────────────────────────────────
def quality_checks(df):
    log.info("Running data quality checks...")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        log.info("  ✅ Zero missing values")
    else:
        log.warning(f"  ⚠️  Missing values:\n{null_counts[null_counts > 0]}")

    dups = df.duplicated(subset=["district", "date"]).sum()
    if dups == 0:
        log.info("  ✅ Zero duplicate district+date combinations")
    else:
        log.warning(f"  ⚠️  {dups} duplicate district+date rows found!")

    log.info(f"  Districts: {df['district'].nunique()}")
    log.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"  Total rows: {len(df):,}")


# ── Step 8: Create Binary Classification Target ───────────────────────────────
def create_targets(df, cfg):
    """
    Classification target : rain_today = 1 if PRECTOTCORR > 2.5mm, else 0
    Regression target     : PRECTOTCORR (already in df, optionally log-transformed)
    """
    threshold = cfg["target"]["rain_threshold_mm"]
    df = df.copy()
    df["rain_today"] = (df["PRECTOTCORR"] > threshold).astype(int)

    rain_pct = df["rain_today"].mean() * 100
    ratio    = (1 - df["rain_today"].mean()) / df["rain_today"].mean()
    log.info(f"Target created — Rain days: {rain_pct:.1f}%, Imbalance ratio: 1:{ratio:.1f}")

    return df


# ── Step 9: Temporal Train / Val / Test Split ─────────────────────────────────
def temporal_split(df, cfg):
    """
    Temporal (chronological) split — never shuffle.
    Shuffling would allow future data to predict the past → data leakage.

    Train : 2017-01-01 → 2022-12-31
    Val   : 2023-01-01 → 2023-12-31
    Test  : 2024-01-01 → 2025-11-30
    """
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end   = pd.Timestamp(cfg["split"]["val_end"])

    train = df[df["date"] <= train_end].copy()
    val   = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test  = df[df["date"] > val_end].copy()

    log.info(f"Temporal split:")
    log.info(f"  Train : {train['date'].min().date()} → {train['date'].max().date()}  ({len(train):,} rows)")
    log.info(f"  Val   : {val['date'].min().date()} → {val['date'].max().date()}    ({len(val):,} rows)")
    log.info(f"  Test  : {test['date'].min().date()} → {test['date'].max().date()}  ({len(test):,} rows)")

    return train, val, test


# ── Step 10: Save Outputs ─────────────────────────────────────────────────────
def save_outputs(df, train, val, test, cfg):
    os.makedirs(os.path.dirname(cfg["paths"]["processed"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["train"]),     exist_ok=True)
    df.to_csv(cfg["paths"]["processed"], index=False)
    log.info(f"Saved processed dataset → {cfg['paths']['processed']}")

    train.to_csv(cfg["paths"]["train"], index=False)
    val.to_csv(cfg["paths"]["val"],     index=False)
    test.to_csv(cfg["paths"]["test"],   index=False)
    log.info(f"Saved splits → data/splits/")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing(config_path="configs/config.yaml"):
    cfg = load_config(config_path)

    era5, nasa, coords = load_raw(cfg)
    era5, nasa, coords = normalize_districts(era5, nasa, coords)
    nasa   = drop_kanpur_duplicate(nasa, cfg)
    era5   = convert_era5_units(era5)
    nasa   = winsorize_nasa_precip(nasa, cfg)
    merged = merge_datasets(era5, nasa, coords)
    quality_checks(merged)
    merged = create_targets(merged, cfg)
    train, val, test = temporal_split(merged, cfg)
    save_outputs(merged, train, val, test, cfg)

    log.info("✅ Preprocessing complete.")
    return merged, train, val, test


if __name__ == "__main__":
    run_preprocessing()
