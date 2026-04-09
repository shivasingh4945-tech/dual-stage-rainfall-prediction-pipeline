"""
visualize.py
────────────
All visualizations — EDA, preprocessing, model evaluation.
Every plot is saved to outputs/plots/.

Usage:
    python src/visualize.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import logging
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score
)

from config_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

plt.style.use("dark_background")
CMAP_MAIN    = "#4fc3f7"
CMAP_ACCENT  = "#ff9800"
CMAP_DANGER  = "#ef5350"
CMAP_SUCCESS = "#4caf50"
CMAP_PURPLE  = "#ab47bc"
BG           = "#0e1117"
BG2          = "#141c2f"
def savefig(name, cfg):
    os.makedirs(cfg["paths"]["plots"], exist_ok=True)
    path = os.path.join(cfg["paths"]["plots"], f"{name}.png")
    plt.savefig(path, dpi=cfg["viz"]["dpi"], bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"  Saved → {path}")


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    ax.set_title(title, color=CMAP_MAIN, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="#90caf9", fontsize=9)
    ax.set_ylabel(ylabel, color="#90caf9", fontsize=9)
    ax.tick_params(colors="#78909c", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1565c0")


# ════════════════════════════════════════════════════════════════
#  EDA PLOTS
# ════════════════════════════════════════════════════════════════

def plot_target_distribution(df, cfg):
    """Rain/No Rain class balance + precipitation distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

    # Histogram of PRECTOTCORR > 0
    ax = axes[0]
    ax.set_facecolor(BG2)
    nonzero = df[df["PRECTOTCORR"] > 0]["PRECTOTCORR"]
    ax.hist(nonzero, bins=80, color=CMAP_MAIN, alpha=0.8, edgecolor="none")
    ax.axvline(2.5, color=CMAP_DANGER, lw=2, linestyle="--", label="2.5mm IMD threshold")
    ax.legend(fontsize=8)
    style_ax(ax, "Precipitation Distribution (>0 mm)", "mm/day", "Count")

    # Log scale
    ax = axes[1]
    ax.set_facecolor(BG2)
    ax.hist(np.log1p(df["PRECTOTCORR"]), bins=80, color=CMAP_ACCENT, alpha=0.8, edgecolor="none")
    style_ax(ax, "log(1 + Precipitation) Distribution", "log(1+mm)", "Count")

    # Pie chart class balance
    ax = axes[2]
    ax.set_facecolor(BG2)
    vc = df["rain_today"].value_counts()
    wedges, texts, autotexts = ax.pie(
        vc.values,
        labels=["No Rain", "Rain"],
        autopct="%1.1f%%",
        colors=[CMAP_MAIN, CMAP_DANGER],
        startangle=90,
        explode=[0, 0.05],
    )
    for t in texts + autotexts:
        t.set_color("#e0e0e0")
        t.set_fontsize(10)
    style_ax(ax, "Class Balance: Rain vs No Rain", "", "")

    plt.suptitle("Target Variable Analysis", color=CMAP_MAIN, fontsize=13)
    plt.tight_layout()
    savefig("01_target_distribution", cfg)


def plot_monthly_rainfall(df, cfg):
    """Monthly mean precipitation and rain day frequency."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    colors = [CMAP_ACCENT if m in [6,7,8,9] else CMAP_MAIN for m in range(1, 13)]

    monthly_mean = df.groupby("month")["PRECTOTCORR"].mean()
    rain_freq    = df.groupby("month")["rain_today"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    axes[0].set_facecolor(BG2)
    axes[0].bar(month_names, monthly_mean.values, color=colors)
    for i, v in enumerate(monthly_mean.values):
        axes[0].text(i, v + 0.1, f"{v:.1f}", ha="center", color="#e0e0e0", fontsize=8)
    style_ax(axes[0], "Mean Daily Precipitation by Month (NASA POWER)", "Month", "mm/day")

    axes[1].set_facecolor(BG2)
    axes[1].bar(month_names, rain_freq.values, color=colors)
    axes[1].axhline(50, color=CMAP_DANGER, lw=1.5, linestyle="--", alpha=0.7)
    style_ax(axes[1], "Rain Day Frequency by Month (%)", "Month", "% Days with Rain (>2.5mm)")

    plt.suptitle("Seasonal Rainfall Patterns — Uttar Pradesh", color=CMAP_MAIN, fontsize=13)
    plt.tight_layout()
    savefig("02_monthly_rainfall", cfg)


def plot_feature_distributions(df, cfg):
    """Histogram + KDE for all key numeric features."""
    from scipy.stats import gaussian_kde
    key_cols = ["t2m","d2m","sp","u10","v10","RH2M","WS10M","WS50M","PRECTOTCORR","tdd","wind_speed_10m","humidity_stress"]
    key_cols = [c for c in key_cols if c in df.columns]
    n_cols = 4
    n_rows = (len(key_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), facecolor=BG)
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(key_cols):
        ax = axes_flat[i]
        ax.set_facecolor(BG2)
        data = df[col].dropna()
        ax.hist(data, bins=60, color=CMAP_MAIN, alpha=0.7, edgecolor="none", density=True)
        try:
            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 200)
            ax.plot(x, kde(x), color=CMAP_ACCENT, lw=1.5)
        except Exception:
            pass
        ax.axvline(data.mean(),   color=CMAP_DANGER,  lw=1, linestyle="--", alpha=0.8)
        ax.axvline(data.median(), color=CMAP_SUCCESS, lw=1, linestyle="--", alpha=0.8)
        style_ax(ax, col.replace("_"," ").title(), "", "Density")

    for j in range(len(key_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.legend(
        [plt.Line2D([0],[0], color=CMAP_DANGER,  lw=1.5, linestyle="--"),
         plt.Line2D([0],[0], color=CMAP_SUCCESS, lw=1.5, linestyle="--"),
         plt.Line2D([0],[0], color=CMAP_ACCENT,  lw=1.5)],
        ["Mean","Median","KDE"],
        loc="lower right", fontsize=8, labelcolor="#e0e0e0", facecolor=BG2,
    )
    plt.suptitle("Feature Distributions", color=CMAP_MAIN, fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("03_feature_distributions", cfg)


def plot_correlation_heatmap(df, cfg):
    """Pearson correlation heatmap."""
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ["year","day_of_year","rain_today","season_encoded"]
                and "sin" not in c and "cos" not in c][:18]

    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(16, 13), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap="coolwarm", center=0, ax=ax, linewidths=0.3, linecolor=BG,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix", color=CMAP_MAIN, fontsize=13, pad=12)
    ax.tick_params(colors="#90caf9", labelsize=8)
    plt.tight_layout()
    savefig("04_correlation_heatmap", cfg)


def plot_target_correlation(df, cfg):
    """Bar chart of feature correlations with PRECTOTCORR."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()["PRECTOTCORR"].drop("PRECTOTCORR").abs().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG2)
    top = corr.head(20)
    colors = [CMAP_DANGER if v > 0.5 else CMAP_ACCENT if v > 0.3 else CMAP_MAIN for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0.3, color=CMAP_ACCENT, lw=1, linestyle="--", alpha=0.7, label="r=0.3")
    ax.axvline(0.5, color=CMAP_DANGER, lw=1, linestyle="--", alpha=0.7, label="r=0.5")
    ax.legend(fontsize=8)
    style_ax(ax, "Feature Correlation with Precipitation (|Pearson r|)", "|r|", "Feature")
    plt.tight_layout()
    savefig("05_target_correlation", cfg)


def plot_outlier_boxplots(df, cfg):
    """Box plots for outlier detection."""
    key_cols = ["PRECTOTCORR","t2m","d2m","sp","RH2M","WS50M","wind_speed_10m","tdd"]
    key_cols = [c for c in key_cols if c in df.columns]
    n_cols = 4
    n_rows = (len(key_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), facecolor=BG)
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(key_cols):
        ax = axes_flat[i]
        ax.set_facecolor(BG2)
        data = df[col].dropna()
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#1a2744", color=CMAP_MAIN),
                   whiskerprops=dict(color="#90caf9"),
                   medianprops=dict(color=CMAP_ACCENT, lw=2),
                   flierprops=dict(marker="o", markerfacecolor=CMAP_DANGER, markersize=2, alpha=0.4))
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        n_out = ((data < q1 - 1.5*(q3-q1)) | (data > q3 + 1.5*(q3-q1))).sum()
        style_ax(ax, f"{col}\n({n_out:,} outliers)", "", "Value")

    for j in range(len(key_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Outlier Detection — IQR Method", color=CMAP_MAIN, fontsize=13, y=1.01)
    plt.tight_layout()
    savefig("06_outlier_boxplots", cfg)


def plot_temporal_trends(df, cfg):
    """Annual trend, monthly box plot, seasonal contribution, rolling mean."""
    df = df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=BG)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Annual trend
    ax = axes[0,0]; ax.set_facecolor(BG2)
    annual = df.groupby("year")["PRECTOTCORR"].mean()
    ax.bar(annual.index, annual.values, color=CMAP_MAIN, alpha=0.8)
    z = np.polyfit(annual.index, annual.values, 1)
    ax.plot(annual.index, np.poly1d(z)(annual.index), "--", color=CMAP_DANGER, lw=2, label="Trend")
    ax.legend(fontsize=8)
    style_ax(ax, "Annual Mean Precipitation (All 72 Districts)", "Year", "mm/day")

    # Monthly box plots
    ax = axes[0,1]; ax.set_facecolor(BG2)
    monthly_data = [df[df["month"]==m]["PRECTOTCORR"].dropna().values for m in range(1,13)]
    ax.boxplot(monthly_data, patch_artist=True,
               boxprops=dict(facecolor="#1a2744", color=CMAP_MAIN),
               whiskerprops=dict(color="#90caf9"),
               medianprops=dict(color=CMAP_ACCENT, lw=2),
               flierprops=dict(marker=".", color=CMAP_DANGER, markersize=1, alpha=0.3))
    ax.set_xticklabels(month_names, fontsize=8)
    style_ax(ax, "Monthly Precipitation Distribution", "Month", "mm/day")

    # Season contribution
    ax = axes[1,0]; ax.set_facecolor(BG2)
    if "season" in df.columns:
        season_total = df.groupby("season")["PRECTOTCORR"].sum()
        pct = season_total / season_total.sum() * 100
        bars = ax.bar(pct.index, pct.values,
                      color=[CMAP_MAIN, CMAP_ACCENT, CMAP_SUCCESS, CMAP_PURPLE])
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}%", ha="center", color="#e0e0e0", fontsize=9)
        style_ax(ax, "Season-wise % Contribution to Total Rainfall", "Season", "% of Total")

    # Rolling mean for sample district
    ax = axes[1,1]; ax.set_facecolor(BG2)
    sample = df[df["district"] == df["district"].unique()[0]].sort_values("date")
    ax.fill_between(sample["date"], sample["PRECTOTCORR"], alpha=0.3, color=CMAP_MAIN)
    roll = sample["PRECTOTCORR"].rolling(30).mean()
    ax.plot(sample["date"], roll, color=CMAP_ACCENT, lw=1.5, label="30d Rolling Mean")
    ax.legend(fontsize=8)
    style_ax(ax, f"Daily Precip — {df['district'].unique()[0].replace('_',' ').title()}", "Date", "mm/day")

    plt.suptitle("Temporal Trends", color=CMAP_MAIN, fontsize=13)
    plt.tight_layout()
    savefig("07_temporal_trends", cfg)


def plot_spatial_patterns(df, cfg):
    """Lon/Lat vs precipitation scatter."""
    if "latitude" not in df.columns:
        return

    dist_stats = df.groupby("district").agg(
        mean_precip=("PRECTOTCORR","mean"),
        latitude=("latitude","first"),
        longitude=("longitude","first"),
        rain_pct=("rain_today","mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax, xcol, xlabel in zip(axes, ["longitude","latitude"], ["Longitude (°E)","Latitude (°N)"]):
        ax.set_facecolor(BG2)
        sc = ax.scatter(dist_stats[xcol], dist_stats["mean_precip"],
                        c=dist_stats["mean_precip"], cmap="Blues", s=70,
                        alpha=0.9, edgecolors="#1565c0", lw=0.5)
        plt.colorbar(sc, ax=ax, label="mm/day")
        z = np.polyfit(dist_stats[xcol], dist_stats["mean_precip"], 1)
        xline = np.linspace(dist_stats[xcol].min(), dist_stats[xcol].max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), "--", color=CMAP_ACCENT, lw=2)
        style_ax(ax, f"{xlabel} vs Mean Precipitation", xlabel, "Mean Precip (mm/day)")

    plt.suptitle("Spatial Rainfall Gradient — Uttar Pradesh", color=CMAP_MAIN, fontsize=13)
    plt.tight_layout()
    savefig("08_spatial_patterns", cfg)


# ════════════════════════════════════════════════════════════════
#  MODEL EVALUATION PLOTS
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, cfg):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Rain","Rain"],
                yticklabels=["No Rain","Rain"],
                annot_kws={"size": 12})
    ax.set_xlabel("Predicted", color="#90caf9", fontsize=10)
    ax.set_ylabel("Actual", color="#90caf9", fontsize=10)
    ax.set_title("Confusion Matrix", color=CMAP_MAIN, fontsize=12)
    ax.tick_params(colors="#90caf9")
    plt.tight_layout()
    savefig("09_confusion_matrix", cfg)


def plot_roc_curve(y_true, y_proba, cfg):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG2)
    ax.plot(fpr, tpr, color=CMAP_MAIN, lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0,1],[0,1], "--", color="#78909c", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color=CMAP_MAIN)
    ax.legend(fontsize=9)
    style_ax(ax, "ROC Curve — Rain/No Rain Classifier", "False Positive Rate", "True Positive Rate")
    plt.tight_layout()
    savefig("10_roc_curve", cfg)


def plot_precision_recall(y_true, y_proba, cfg):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG2)
    ax.plot(rec, prec, color=CMAP_ACCENT, lw=2, label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y_true.mean(), color="#78909c", lw=1, linestyle="--", label="Baseline (class freq)")
    ax.fill_between(rec, prec, alpha=0.1, color=CMAP_ACCENT)
    ax.legend(fontsize=9)
    style_ax(ax, "Precision-Recall Curve", "Recall", "Precision")
    plt.tight_layout()
    savefig("11_precision_recall_curve", cfg)


def plot_feature_importance(feature_names, importances, model_name, cfg, top_n=20):
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG2)
    colors = [CMAP_DANGER if v > fi.max()*0.7 else CMAP_ACCENT if v > fi.max()*0.4 else CMAP_MAIN
              for v in fi.values]
    ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1])
    style_ax(ax, f"Top {top_n} Feature Importances — {model_name}", "Importance Score", "Feature")
    plt.tight_layout()
    savefig(f"12_feature_importance_{model_name.lower().replace(' ','_')}", cfg)


def plot_actual_vs_predicted(y_true, y_pred, model_name, cfg):
    """Scatter plot of actual vs predicted rainfall (rainy days only)."""
    mask = y_true > 2.5
    yt, yp = y_true[mask], y_pred[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # Scatter
    ax = axes[0]; ax.set_facecolor(BG2)
    ax.scatter(yt, yp, alpha=0.3, s=10, color=CMAP_MAIN, edgecolors="none")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "--", color=CMAP_DANGER, lw=1.5, label="Perfect prediction")
    ax.legend(fontsize=8)
    style_ax(ax, f"Actual vs Predicted — {model_name}\n(Rainy days >2.5mm only)",
             "Actual (mm)", "Predicted (mm)")

    # Residuals
    ax = axes[1]; ax.set_facecolor(BG2)
    residuals = yp - yt
    ax.scatter(yt, residuals, alpha=0.3, s=10, color=CMAP_ACCENT, edgecolors="none")
    ax.axhline(0, color=CMAP_DANGER, lw=1.5, linestyle="--")
    style_ax(ax, f"Residuals — {model_name}", "Actual (mm)", "Residual (Predicted - Actual)")

    plt.suptitle(f"Regression Diagnostics — {model_name}", color=CMAP_MAIN, fontsize=13)
    plt.tight_layout()
    savefig(f"13_actual_vs_predicted_{model_name.lower().replace(' ','_')}", cfg)


def run_all_eda_plots(config_path="configs/config.yaml"):
    """Run all EDA visualizations on the processed dataset."""
    cfg = load_config(config_path)
    os.makedirs(cfg["paths"]["plots"], exist_ok=True)

    df = pd.read_csv(cfg["paths"]["processed"], parse_dates=["date"])

    # Add derived columns needed for plots
    from features import add_temporal_features, add_wind_features, add_thermo_features
    df = add_temporal_features(df)
    df = add_wind_features(df)
    df = add_thermo_features(df)

    log.info("Generating EDA plots...")
    plot_target_distribution(df, cfg)
    plot_monthly_rainfall(df, cfg)
    plot_feature_distributions(df, cfg)
    plot_correlation_heatmap(df, cfg)
    plot_target_correlation(df, cfg)
    plot_outlier_boxplots(df, cfg)
    plot_temporal_trends(df, cfg)
    plot_spatial_patterns(df, cfg)
    log.info("✅ All EDA plots saved.")


if __name__ == "__main__":
    run_all_eda_plots()
