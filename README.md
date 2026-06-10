# Rainfall Prediction — Uttar Pradesh (2017–2025)

District-level daily rainfall prediction for 72 UP districts using ERA5 reanalysis + NASA POWER,
XGBoost + LightGBM models, temporal cross-validation, and SHAP explainability.

---

## Project Structure

```
rainfall_prediction/
│
├── data/
│   ├── raw/                         ← Original datasets (do not edit)
│   │   ├── era5_district_daily_2017_2025.csv
│   │   ├── nasa_power_combined.csv
│   │   └── district_coordinates.csv
│   ├── processed/
│   │   └── merged_processed.csv     ← Merged + cleaned + unit-converted
│   └── splits/
│       ├── train.csv                ← 2017–2022
│       ├── val.csv                  ← 2023
│       └── test.csv                 ← 2024–2025
│
├── notebooks/
│   ├── 01_eda.ipynb                 ← Data examination, distributions, correlations
│   ├── 02_preprocessing.ipynb       ← Pipeline walkthrough step by step
│   ├── 03_classification.ipynb      ← Rain/No Rain XGBoost training + eval
│   ├── 04_regression.ipynb          ← Rainfall amount XGBoost + LightGBM
│   └── 05_visualizations.ipynb      ← Regenerate all plots
│
├── src/
│   ├── preprocess.py                ← Full preprocessing pipeline
│   ├── features.py                  ← All feature engineering
│   ├── train_classifier.py          ← XGBoost classifier
│   ├── train_regressor.py           ← XGBoost + LightGBM regressor
│   ├── evaluate.py                  ← Comprehensive evaluation
│   └── visualize.py                 ← All plots (EDA + model)
│
├── models/
│   ├── classifier_xgb.pkl
│   ├── regressor_xgb.pkl
│   ├── regressor_lgbm.pkl
│   ├── scaler.pkl
│   └── scaler_reg.pkl
│
├── outputs/
│   ├── plots/                       ← All saved figures (13+ plots)
│   ├── reports/                     ← Metrics JSON + feature importance CSV
│   └── predictions.csv              ← Test set predictions
│
├── configs/
│   └── config.yaml                  ← All hyperparameters and paths
│
├── requirements.txt
└── README.md
```

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess — merge datasets, engineer features, split
python src/preprocess.py

# 3. Train classifier (Rain / No Rain)
python src/train_classifier.py

# 4. Train regressor (rainfall amount mm/day)
python src/train_regressor.py

# 5. Full evaluation report
python src/evaluate.py

# 6. Generate all plots
python src/visualize.py

# Or use Jupyter notebooks (recommended)
jupyter lab
```

---

## Docker

Build the image and run the full pipeline:

```bash
docker compose up --build
```

Run only selected stages:

```bash
docker compose run --rm rainfall-pipeline preprocess classifier
docker compose run --rm rainfall-pipeline evaluate
```

The container uses `run_pipeline.py` as its entrypoint, so stage names passed after the
service name are forwarded directly to the pipeline runner. `configs/`, `data/`,
`models/`, and `outputs/` are mounted from the host machine.

---

## Dataset Summary

| Dataset | Rows | Columns | Key Contents |
|---|---|---|---|
| ERA5 | 234,432 | 8 | t2m, d2m, sp, u10, v10, tp — 72 districts |
| NASA POWER | 237,688 | 10 | PRECTOTCORR, RH2M, WS50M, WD10M, WD50M |
| Coordinates | 72 | 3 | Latitude, Longitude per district |
| **Merged** | **234,432** | **18** | All combined |

---

## 🎯 Target Variables

| Task | Target | Source | Notes |
|---|---|---|---|
| Classification | `rain_today` | Derived | PRECTOTCORR > 2.5mm → 1, else 0 |
| Regression | `PRECTOTCORR` | NASA POWER | log1p transformed, expm1 at eval |

**Why NASA PRECTOTCORR and not ERA5 tp?**
ERA5 `tp` after conversion gives 0.44 mm/day for Lucknow in July — unrealistically low.
NASA `PRECTOTCORR` gives 11.35 mm/day for the same — matches IMD gauge data.

---

## ⚠️ Key Data Issues Found & Fixed

| Issue | Fix |
|---|---|
| ERA5 columns in Kelvin/Pa/metres | Converted to °C/hPa/mm |
| NASA `kanpur` duplicate (3,256 rows) | Dropped — kept `kanpur_nagar` |
| NASA 279mm February event (artifact) | Winsorized at 99th pct (~38mm) |
| Coords name mismatch (spaces vs underscores) | Normalized before join |
| ERA5 tp unreliable as precip target | Using NASA PRECTOTCORR as target |

---

## 🔧 Feature Engineering

| Feature | Source | Why |
|---|---|---|
| `month_sin`, `month_cos` | date | Circular month — Dec adjacent to Jan |
| `season_encoded` | month | 4 Indian met seasons |
| `wind_speed_10m` | u10, v10 | √(U²+V²) scalar wind |
| `wind_dir_sin/cos` | u10, v10 | Circular wind direction |
| `wd10m_sin/cos`, `wd50m_sin/cos` | WD10M, WD50M | NASA circular wind direction |
| `tdd` | t2m, d2m | T - Td depression (< 3°C → rain likely) |
| `humidity_stress` | RH2M, tdd | RH / TDD — moisture saturation index |
| `precip_lag_1d/3d/7d` | PRECTOTCORR | Rainfall persistence (r≈0.55 at lag-1) |
| `precip_roll_7d/30d` | PRECTOTCORR | Active/break monsoon signal |
| `rain_streak` | rain_today | Consecutive rain day count |

---

## 📈 Train / Val / Test Split

| Split | Period | Rows (approx) | Purpose |
|---|---|---|---|
| Train | 2017–2022 | ~168k | Model training + SMOTE |
| Val | 2023 | ~26k | Hyperparameter selection |
| Test | 2024–2025 | ~41k | Final unbiased evaluation |

**Temporal split is mandatory** — shuffling would allow future data to predict the past (data leakage).

---

## 🤖 Models

**Classification — XGBoost**
- Scale pos weight = 3.13 (imbalance ratio)
- SMOTE on training set only
- Metrics: Accuracy, F1, ROC-AUC, Average Precision

**Regression — XGBoost + LightGBM (ensembled)**
- log1p target transform → expm1 at evaluation
- Metrics: RMSE, MAE, R² (all days + rainy days only)
- Simple average ensemble of XGB and LGBM predictions
