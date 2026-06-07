# =============================================================================
#  CREDIT CARD FRAUD DETECTION — DAY 3
#  Threshold Tuning  |  SHAP Analysis  |  Business Impact
#
#  Inputs (from previous days):
#    ../DAY 2/best_model.pkl               <- Fitted XGBoost classifier
#    ../DAY 1/processed/test_raw.parquet   <- 56,746-row held-out test set
#    ../DAY 1/processed/train_raw.parquet  <- training split (not resampled)
#
#  Outputs written to ./output/:
#    plots/01_threshold_metrics.png         P/R/F1/F2 vs threshold + optimal markers
#    plots/02_threshold_cost_curve.png      FP/FN/Total cost vs threshold
#    plots/03_threshold_confusion_compare.png  Default 0.50 vs optimal side-by-side
#    plots/04_shap_global_bar.png           Top-20 mean |SHAP| bar chart
#    plots/05_shap_beeswarm.png             Custom beeswarm — top-15 features
#    plots/06_shap_waterfall_fraud.png      Waterfall for highest-confidence fraud
#    plots/07_shap_dependence_top5.png      Dependence scatter for top-5 features
#    plots/08_business_impact_scenarios.png Annual cost + fraud catch rate bars
#    plots/09_cost_sensitivity_heatmap.png  Sensitivity to FP/FN cost assumptions
#    results/threshold_analysis.csv
#    results/shap_top5_features.csv
#    results/business_impact_report.txt
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, fbeta_score, accuracy_score, confusion_matrix,
)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[WARN] shap not installed.  Run: python -m pip install shap")


# Sigmoid (no hard scipy dependency)
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


# =============================================================================
#  CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# -- Paths (relative to DAY 2 directory) --------------------------------------
MODEL_PATH          = "best_model.pkl"
TRAIN_PATH          = os.path.join("..", "DAY 1", "processed", "train_raw.parquet")
TEST_PATH           = os.path.join("..", "DAY 1", "processed", "test_raw.parquet")
# Validation set saved by fraud_detection_day2.py — used for threshold tuning ONLY
VAL_X_PATH          = os.path.join("output", "results", "validation_X.parquet")
VAL_Y_PATH          = os.path.join("output", "results", "validation_y.parquet")
BEST_THRESHOLD_PATH = "best_threshold.pkl"
TARGET_COL          = "Class"

# -- Output directories -------------------------------------------------------
OUTPUT_DIR  = "output"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# -- Threshold sweep ----------------------------------------------------------
THRESH_MIN  = 0.10
THRESH_MAX  = 0.90
THRESH_STEP = 0.01   # 81 thresholds -> smooth curves

# -- Business cost parameters (USD) ------------------------------------------
#    Computed dynamically from test-set fraud amounts; override here if needed
COST_FP_USD           = 50.0   # investigation + analyst time + customer friction
CHARGEBACK_MULTIPLIER = 2.5    # fraud loss x chargeback/fee multiplier

# -- Annual volume parameters -------------------------------------------------
DAYS_IN_DATASET = 2.0   # creditcard.csv spans ~48 hours (Time column max ~172800 s)
TRAIN_FRAC      = 0.80  # train/test split ratio used in Day 1

# -- SHAP visualisation -------------------------------------------------------
SHAP_PLOT_SAMPLE = 2_000   # subsample for beeswarm / dependence scatter
TOP_N_SHAP_BAR   = 20      # features shown in global importance bar chart
TOP_N_BEESWARM   = 15      # features shown in beeswarm

# -- Dark colour palette (mirrors Day 1 & 2) ----------------------------------
CHARCOAL     = "#1C1C2E"
PANEL_BG     = "#252540"
GRID_COL     = "#3A3A5C"
TEXT_COL     = "#E0E0F0"
COLOR_PREC   = "#00D4AA"   # teal   - Precision
COLOR_REC    = "#FF4C61"   # red    - Recall
COLOR_F1     = "#7B68EE"   # purple - F1
COLOR_F2     = "#FFD700"   # gold   - F2
COLOR_COST   = "#FF8C42"   # orange - Cost curves
COLOR_XGB    = "#FF4C61"   # red    - best model (XGBoost)

plt.rcParams.update({
    "figure.facecolor": CHARCOAL, "axes.facecolor":  PANEL_BG,
    "axes.edgecolor":   GRID_COL,  "axes.labelcolor": TEXT_COL,
    "axes.titlecolor":  TEXT_COL,  "xtick.color":     TEXT_COL,
    "ytick.color":      TEXT_COL,  "text.color":      TEXT_COL,
    "grid.color":       GRID_COL,  "grid.linestyle":  "--",
    "grid.alpha":       0.4,       "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COL,  "font.family":     "DejaVu Sans",
})


# =============================================================================
#  STEP 1 — LOAD MODEL & DATA
# =============================================================================

def load_artifacts():
    """
    Load the best model, test set, and train set.
    Computes predicted probabilities and derives business cost parameters
    dynamically from the actual test-set fraud amounts.
    """
    print("\n" + "=" * 68)
    print("  STEP 1 — LOAD MODEL & DATA")
    print("=" * 68)

    # Model
    assert os.path.exists(MODEL_PATH), \
        f"Model not found: {os.path.abspath(MODEL_PATH)}"
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    print(f"  Model type   : {type(model).__name__}")
    print(f"  n_estimators : {getattr(model, 'n_estimators', '?')}")

    # Test set
    assert os.path.exists(TEST_PATH), \
        f"Test file not found: {os.path.abspath(TEST_PATH)}"
    test_df = pd.read_parquet(TEST_PATH)
    feat_cols = [c for c in test_df.columns if c != TARGET_COL]
    X_test  = test_df[feat_cols].copy()
    y_test  = test_df[TARGET_COL].copy()

    # Predicted probabilities on test (used for final evaluation only)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Validation set — carved from training data in Day 2; used for threshold tuning
    assert os.path.exists(VAL_X_PATH), (
        f"Validation X not found: {os.path.abspath(VAL_X_PATH)}\n"
        f"  Run fraud_detection_day2.py first to generate validation parquets."
    )
    assert os.path.exists(VAL_Y_PATH), (
        f"Validation y not found: {os.path.abspath(VAL_Y_PATH)}\n"
        f"  Run fraud_detection_day2.py first to generate validation parquets."
    )
    X_val      = pd.read_parquet(VAL_X_PATH)
    y_val      = pd.read_parquet(VAL_Y_PATH).iloc[:, 0]   # single-column DataFrame -> Series
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Train set (for SHAP background context)
    assert os.path.exists(TRAIN_PATH), \
        f"Train file not found: {os.path.abspath(TRAIN_PATH)}"
    train_df = pd.read_parquet(TRAIN_PATH)
    X_train  = train_df[feat_cols].copy()

    # Business cost: FN cost derived from actual fraud amounts in test set
    avg_fraud_amt = test_df.loc[y_test == 1, "Amount"].mean()
    cost_fn_usd   = round(avg_fraud_amt * CHARGEBACK_MULTIPLIER, 2)

    # Annual scale factor
    total_rows_est = len(y_test) / (1.0 - TRAIN_FRAC)
    annual_scale   = (total_rows_est / DAYS_IN_DATASET) * 365.25

    # Log
    n_fraud = int(y_test.sum())
    print(f"\n  Test set      : {len(X_test):,} rows | "
          f"fraud={n_fraud:,} ({y_test.mean()*100:.4f}%) | "
          f"legit={len(y_test)-n_fraud:,}  [final evaluation only]")
    print(f"  Val set       : {len(X_val):,} rows | "
          f"fraud={int(y_val.sum()):,} ({y_val.mean()*100:.4f}%)  "
          f"[threshold tuning only]")
    print(f"  Train set     : {len(X_train):,} rows (for SHAP context)")
    print(f"  Features      : {len(feat_cols)}")
    print(f"\n  Cost per FP   : ${COST_FP_USD:.2f}  "
          f"(investigation + customer friction)")
    print(f"  Cost per FN   : ${cost_fn_usd:.2f}  "
          f"(avg fraud ${avg_fraud_amt:.2f} x {CHARGEBACK_MULTIPLIER}x chargeback)")
    print(f"  Annual volume : ~{annual_scale:,.0f} transactions/yr")
    print("  Load: [OK]")

    return (model, X_test, y_test, X_val, y_val, y_val_prob, X_train,
            y_prob, feat_cols, avg_fraud_amt, cost_fn_usd, annual_scale)


# =============================================================================
#  STEP 2 — THRESHOLD ANALYSIS
# =============================================================================

def compute_threshold_metrics(y_test: pd.Series,
                               y_prob: np.ndarray,
                               cost_fn_usd: float,
                               amount_series: pd.Series) -> pd.DataFrame:
    """
    Sweep thresholds from THRESH_MIN to THRESH_MAX (step THRESH_STEP).
    For each threshold compute classification metrics and business costs.
    FN cost = actual sum of missed-fraud transaction amounts × CHARGEBACK_MULTIPLIER
    (transaction-level; more accurate than fn_count × average_cost).
    Returns a DataFrame with one row per threshold.
    """
    thresholds = np.round(np.arange(THRESH_MIN, THRESH_MAX + 1e-9, THRESH_STEP), 4)
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        prec     = precision_score(y_test, y_pred, zero_division=0)
        rec      = recall_score(y_test, y_pred, zero_division=0)
        f1       = f1_score(y_test, y_pred, zero_division=0)
        f2       = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        tpr      = tp / max(tp + fn, 1)
        fpr      = fp / max(fp + tn, 1)
        youden_j = tpr - fpr
        fp_cost  = fp * COST_FP_USD
        # Transaction-level FN cost: sum actual amounts of missed fraud × chargeback rate
        fn_mask  = (y_test.values == 1) & (y_pred == 0)
        fn_cost  = float(amount_series.values[fn_mask].sum()) * CHARGEBACK_MULTIPLIER
        tot_cost = fp_cost + fn_cost

        rows.append({
            "threshold":       t,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision":       round(prec, 6),
            "Recall":          round(rec, 6),
            "F1":              round(f1, 6),
            "F2":              round(f2, 6),
            "TPR":             round(tpr, 6),
            "FPR":             round(fpr, 6),
            "Youden_J":        round(youden_j, 6),
            "Cost_FP_USD":     round(fp_cost, 2),
            "Cost_FN_USD":     round(fn_cost, 2),
            "Total_Cost_USD":  round(tot_cost, 2),
        })

    return pd.DataFrame(rows)


def find_optimal_thresholds(df: pd.DataFrame) -> dict:
    """
    Locate four optimal thresholds and return as {name: Series} dict.
    Primary business recommendation: minimise total cost (FP + FN).
    """
    return {
        "F1-Optimal":              df.loc[df["F1"].idxmax()],
        "F2-Optimal (Recall x2)":  df.loc[df["F2"].idxmax()],
        "Cost-Optimal (Business)": df.loc[df["Total_Cost_USD"].idxmin()],
        "Youden-J (Balanced)":     df.loc[df["Youden_J"].idxmax()],
    }


def print_threshold_summary(df: pd.DataFrame,
                              opt: dict,
                              cost_fn_usd: float):
    """Pretty-print threshold summary to stdout."""
    print("\n" + "=" * 68)
    print("  THRESHOLD OPTIMISATION SUMMARY")
    print("=" * 68)
    print(f"\n  Cost/FP=${COST_FP_USD:.0f}  "
          f"Cost/FN=${cost_fn_usd:.0f}  "
          f"Thresholds={THRESH_MIN}..{THRESH_MAX} step={THRESH_STEP}\n")

    hdr = (f"  {'Criterion':<28} {'Thresh':>6}  "
           f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'F2':>6}  {'Cost($)':>10}")
    print(hdr)
    print("  " + "-" * 74)

    for name, row in opt.items():
        tag = "  <-- RECOMMENDED" if "Cost" in name else ""
        print(
            f"  {name:<28} {row['threshold']:>6.2f}  "
            f"{row['Precision']:>6.4f}  {row['Recall']:>6.4f}  "
            f"{row['F1']:>6.4f}  {row['F2']:>6.4f}  "
            f"{row['Total_Cost_USD']:>10,.2f}{tag}"
        )

    # Baseline: default 0.50
    dft = df[df["threshold"].between(0.499, 0.501)].iloc[0]
    print("  " + "-" * 74)
    print(
        f"  {'Default (0.50)':<28} {dft['threshold']:>6.2f}  "
        f"{dft['Precision']:>6.4f}  {dft['Recall']:>6.4f}  "
        f"{dft['F1']:>6.4f}  {dft['F2']:>6.4f}  "
        f"{dft['Total_Cost_USD']:>10,.2f}  <- baseline"
    )


# ── Threshold plots ─────────────────────────────────────────────────────────

def plot_threshold_metrics(df: pd.DataFrame,
                            opt: dict,
                            save_path: str):
    """
    Two-panel figure:
      Top   : Precision / Recall / F1 / F2 curves with optimal-threshold markers.
      Bottom: TP, FP, FN count curves (shows trade-off concretely).
    """
    opt_colors = {
        "F1-Optimal":              ("#7B68EE", "F1"),
        "F2-Optimal (Recall x2)":  ("#FFD700", "F2"),
        "Cost-Optimal (Business)": ("#FF8C42", "Cost"),
        "Youden-J (Balanced)":     ("#00BFFF", "Youden"),
    }

    fig = plt.figure(figsize=(13, 10), facecolor=CHARCOAL)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.set_facecolor(PANEL_BG)
    ax2.set_facecolor(PANEL_BG)

    t = df["threshold"].values

    # --- Top panel: metric lines ---
    ax1.plot(t, df["Precision"], color=COLOR_PREC, lw=2.2, label="Precision", zorder=3)
    ax1.plot(t, df["Recall"],    color=COLOR_REC,  lw=2.2, label="Recall",    zorder=3)
    ax1.plot(t, df["F1"],        color=COLOR_F1,   lw=2.2, label="F1",        zorder=3)
    ax1.plot(t, df["F2"],        color=COLOR_F2,   lw=2.2, label="F2 (beta=2)",
             linestyle="--", zorder=3)

    # Shade region under F2 curve lightly
    ax1.fill_between(t, df["F2"].values, alpha=0.07, color=COLOR_F2)

    for name, row in opt.items():
        c, lbl = opt_colors[name]
        ax1.axvline(row["threshold"], color=c, lw=1.6, linestyle=":",
                    alpha=0.9, zorder=4)
        ypos = {"F1": 0.58, "F2": 0.47, "Cost": 0.36, "Youden": 0.25}.get(lbl, 0.15)
        ax1.text(row["threshold"] + 0.006, ypos,
                 f"{lbl}\n{row['threshold']:.2f}",
                 color=c, fontsize=8, rotation=90, va="bottom")

    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Classification Metrics vs Decision Threshold",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.legend(fontsize=10, loc="upper right",
               framealpha=0.85, edgecolor=GRID_COL)
    ax1.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- Bottom panel: raw counts ---
    ax2.plot(t, df["TP"], color=COLOR_PREC, lw=2.0,
             label="TP (fraud caught)")
    ax2.plot(t, df["FP"], color=COLOR_F2,   lw=2.0,
             label="FP (false alarms)", linestyle="--")
    ax2.plot(t, df["FN"], color=COLOR_REC,  lw=2.0,
             label="FN (missed fraud)", linestyle=":")

    for name, row in opt.items():
        c, _ = opt_colors[name]
        ax2.axvline(row["threshold"], color=c, lw=1.2,
                    linestyle=":", alpha=0.7)

    ax2.set_xlabel("Decision Threshold", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.legend(fontsize=8.5, loc="upper right",
               framealpha=0.85, edgecolor=GRID_COL)
    ax2.grid(True)
    ax2.set_xlim(THRESH_MIN - 0.02, THRESH_MAX + 0.02)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_threshold_cost(df: pd.DataFrame,
                         opt: dict,
                         cost_fn_usd: float,
                         save_path: str):
    """
    Business cost breakdown (FP cost, FN cost, total) vs threshold.
    Annotates the cost-optimal point and the default-0.50 baseline.
    """
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    t = df["threshold"].values

    ax.fill_between(t, df["Cost_FP_USD"], alpha=0.12, color=COLOR_F2)
    ax.fill_between(t, df["Cost_FN_USD"], alpha=0.12, color=COLOR_REC)

    ax.plot(t, df["Cost_FP_USD"],   color=COLOR_F2,  lw=2.0,
            label=f"FP cost  (${COST_FP_USD:.0f} x FP)")
    ax.plot(t, df["Cost_FN_USD"],   color=COLOR_REC, lw=2.0,
            label=f"FN cost  (actual fraud amt x {CHARGEBACK_MULTIPLIER:.1f}x chargeback)")
    ax.plot(t, df["Total_Cost_USD"], color=COLOR_COST, lw=2.8, zorder=5,
            label="Total cost (FP + FN)")

    # Cost-optimal marker
    opt_row = opt["Cost-Optimal (Business)"]
    ax.axvline(opt_row["threshold"], color=COLOR_COST, lw=2.0,
               linestyle="--", alpha=0.85)
    ax.scatter([opt_row["threshold"]], [opt_row["Total_Cost_USD"]],
               color=COLOR_COST, s=120, zorder=6)
    ax.annotate(
        f"  Optimal threshold: {opt_row['threshold']:.2f}\n"
        f"  Min total cost: ${opt_row['Total_Cost_USD']:,.2f}\n"
        f"  TP={opt_row['TP']}  FP={opt_row['FP']}\n"
        f"  FN={opt_row['FN']}  TN={opt_row['TN']:,}",
        xy=(opt_row["threshold"], opt_row["Total_Cost_USD"]),
        xytext=(opt_row["threshold"] + 0.10,
                opt_row["Total_Cost_USD"] + df["Total_Cost_USD"].max() * 0.08),
        fontsize=9, color=TEXT_COL,
        arrowprops=dict(arrowstyle="->", color=TEXT_COL, lw=1.5),
    )

    # Default-0.50 baseline
    dft = df[df["threshold"].between(0.499, 0.501)].iloc[0]
    ax.axvline(0.50, color=GRID_COL, lw=1.5, linestyle=":",
               label=f"Default 0.50  (${dft['Total_Cost_USD']:,.0f})")
    ax.scatter([0.50], [dft["Total_Cost_USD"]], color=GRID_COL, s=80, zorder=5)

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Business Cost (USD) — on Test Set", fontsize=11)
    ax.set_title("Business Cost vs Decision Threshold",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor=GRID_COL)
    ax.grid(True)
    ax.set_xlim(THRESH_MIN - 0.02, THRESH_MAX + 0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_confusion_comparison(y_test: pd.Series,
                               y_prob: np.ndarray,
                               default_thresh: float,
                               opt_thresh: float,
                               save_path: str):
    """
    Side-by-side confusion matrices comparing default threshold vs
    cost-optimal threshold, with metric deltas annotated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=CHARCOAL)

    metrics_dict = {}
    for ax, thresh, title in [
        (axes[0], default_thresh, f"Default Threshold = {default_thresh:.2f}"),
        (axes[1], opt_thresh,     f"Cost-Optimal Threshold = {opt_thresh:.2f}  (Recommended)"),
    ]:
        ax.set_facecolor(PANEL_BG)
        y_pred = (y_prob >= thresh).astype(int)
        cm     = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        f2   = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        metrics_dict[thresh] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                                     prec=prec, rec=rec, f1=f1, f2=f2)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"],
            ax=ax, cbar=False, linewidths=0.5,
            annot_kws={"size": 14, "weight": "bold"},
        )
        ax.set_title(
            f"{title}\n"
            f"Precision={prec:.3f}  Recall={rec:.3f}  "
            f"F1={f1:.3f}  F2={f2:.3f}",
            fontsize=9.5, fontweight="bold", color=TEXT_COL, pad=8,
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)

    # Delta annotation between the two
    m_d = metrics_dict[default_thresh]
    m_o = metrics_dict[opt_thresh]
    delta_recall = m_o["rec"] - m_d["rec"]
    delta_fp     = m_o["fp"]  - m_d["fp"]
    delta_fn     = m_d["fn"]  - m_o["fn"]   # positive = fewer FN (good)

    fig.suptitle(
        f"Confusion Matrix Comparison  |  "
        f"Recall: {m_d['rec']:.3f} -> {m_o['rec']:.3f} ({delta_recall:+.3f})  |  "
        f"FN reduced: {delta_fn:+d}  |  FP change: {delta_fp:+d}",
        fontsize=11, fontweight="bold", color=TEXT_COL, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  STEP 3 — SHAP ANALYSIS
# =============================================================================

def compute_shap_values(model, X_test: pd.DataFrame):
    """
    Compute SHAP values for positive class (fraud) using TreeExplainer.
    Handles both old (list) and new (ndarray) SHAP API output formats.

    Returns:
      shap_vals  : ndarray (n_samples, n_features) — log-odds contributions
      base_value : float  — model expected output (log-odds of prior)
    """
    print("\n  Computing SHAP values for all test samples …",
          end="", flush=True)
    t0 = time.time()

    explainer = shap.TreeExplainer(model)
    raw       = explainer.shap_values(X_test)

    # Normalise to a single 2-D array for positive class
    if isinstance(raw, list):
        shap_vals = raw[1]                              # binary: [neg, pos]
        base_val  = explainer.expected_value
        base_val  = base_val[1] if hasattr(base_val, "__len__") else base_val
    elif raw.ndim == 3:
        shap_vals = raw[:, :, 1]                        # (n, feat, 2)
        base_val  = explainer.expected_value
        base_val  = base_val[1] if hasattr(base_val, "__len__") else base_val
    else:
        shap_vals = raw                                 # (n, feat) — most common
        base_val  = float(np.array(explainer.expected_value).ravel()[-1])

    print(f" done in {time.time()-t0:.1f}s  |  shape={shap_vals.shape}")
    print(f"  Base value (expected log-odds): {float(base_val):.5f}  "
          f"-> P(fraud|prior) = {_sigmoid(base_val):.5f}")

    return shap_vals.astype(float), float(base_val)


def extract_top5_fraud_signals(shap_vals: np.ndarray,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                feat_cols: list) -> pd.DataFrame:
    """
    Identify the top 5 fraud signal features from SHAP values.

    For each feature computes:
      - Mean |SHAP| across ALL test samples (global importance)
      - Mean SHAP across FRAUD-ONLY samples (directional signal on fraud class)
      - Direction: sign of correlation(feature_value, SHAP_value)
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # SHAP values for fraud-positive rows (aligned by positional index)
    fraud_mask = y_test.values.astype(bool)
    mean_fraud  = shap_vals[fraud_mask, :].mean(axis=0)

    top5_idx = np.argsort(mean_abs)[::-1][:5]

    rows = []
    for rank, fi in enumerate(top5_idx, 1):
        fname    = feat_cols[fi]
        fv       = X_test.iloc[:, fi].values.astype(float)
        sv       = shap_vals[:, fi]
        corr     = float(np.corrcoef(fv, sv)[0, 1])
        direction = ("Higher value -> Higher fraud risk"
                     if corr > 0
                     else "Lower value  -> Higher fraud risk")

        # Human-readable description
        if fname.startswith("V"):
            desc = f"PCA component {fname[1:]} (anonymised)"
        elif fname == "Amount":
            desc = "Transaction amount (USD)"
        elif fname == "Time":
            desc = "Seconds elapsed since first transaction"
        elif fname == "hour_of_day":
            desc = "Hour of day (0-24, engineered)"
        elif fname == "amount_log":
            desc = "log(Amount+1) — log-transformed value"
        elif fname == "amount_zscore":
            desc = "Amount z-score (train-set mean/std)"
        else:
            desc = fname

        rows.append({
            "Rank":            rank,
            "Feature":         fname,
            "Mean_Abs_SHAP":   round(float(mean_abs[fi]), 6),
            "Mean_SHAP_Fraud": round(float(mean_fraud[fi]), 6),
            "Corr_SHAP_Feat":  round(corr, 4),
            "Direction":       direction,
            "Description":     desc,
        })

    df5 = pd.DataFrame(rows).set_index("Rank")

    print("\n  TOP 5 FRAUD SIGNAL FEATURES (by mean |SHAP|):")
    print("  " + "-" * 68)
    for rank, row in df5.iterrows():
        print(f"  #{rank}  {row['Feature']:<14}  "
              f"MeanAbsSHAP={row['Mean_Abs_SHAP']:.4f}  "
              f"{row['Direction']}")

    return df5


# ── SHAP plots ───────────────────────────────────────────────────────────────

def plot_shap_global_bar(shap_vals: np.ndarray,
                          feat_cols: list,
                          top_n: int,
                          save_path: str):
    """
    Horizontal bar chart of mean absolute SHAP values.
    Top-5 fraud signals are highlighted in accent colour.
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    series   = pd.Series(mean_abs, index=feat_cols).sort_values()
    top      = series.tail(top_n)

    colors = [
        COLOR_XGB if i >= len(top) - 5 else "#8888BB"
        for i in range(len(top))
    ]

    fig, ax = plt.subplots(figsize=(11, max(7, top_n * 0.44)),
                            facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    bars = ax.barh(range(len(top)), top.values,
                   color=colors, edgecolor=CHARCOAL, alpha=0.87, height=0.72)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9.5)

    x_off = top.values.max() * 0.009
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + x_off,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=8, color=TEXT_COL)

    ax.set_xlabel("Mean |SHAP value|  (avg impact on log-odds)", fontsize=11)
    ax.set_title(f"SHAP Global Feature Importance — Top {top_n}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(right=top.values.max() * 1.20)
    ax.grid(axis="x")
    ax.legend(handles=[
        Patch(color=COLOR_XGB,  label="Top-5 fraud signals"),
        Patch(color="#8888BB",  label="Other features"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_shap_beeswarm(shap_vals: np.ndarray,
                        X_test: pd.DataFrame,
                        feat_cols: list,
                        top_n: int,
                        save_path: str):
    """
    Custom SHAP beeswarm: each dot is one test sample, coloured by
    normalised feature value (blue=low, red=high).
    Jitter on y-axis gives the 'bee' effect without overplotting.
    """
    np.random.seed(RANDOM_STATE)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[-top_n:]         # ascending; last = most important
    n_feat   = len(order)

    # Sub-sample for performance
    n_pts = min(SHAP_PLOT_SAMPLE, len(X_test))
    s_idx = np.random.choice(len(X_test), n_pts, replace=False)

    fig, ax = plt.subplots(figsize=(12, max(8, n_feat * 0.52)), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    cmap = plt.cm.RdBu_r    # red = high feature value, blue = low
    last_sc = None

    for plot_pos, fi in enumerate(order):
        shap_col = shap_vals[s_idx, fi]
        feat_col = X_test.iloc[s_idx, fi].values.astype(float)

        # Normalise feature value for colour
        lo, hi    = np.percentile(feat_col, [1, 99])
        feat_norm = np.clip((feat_col - lo) / max(hi - lo, 1e-9), 0, 1)

        # Jitter y
        y_j = plot_pos + np.random.uniform(-0.38, 0.38, n_pts)

        last_sc = ax.scatter(
            shap_col, y_j,
            c=feat_norm, cmap=cmap, alpha=0.55, s=7,
            vmin=0, vmax=1, rasterized=True,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("Feature value  (blue=low, red=high)",
                 color=TEXT_COL, fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL, labelcolor=TEXT_COL)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feat_cols[i] for i in order], fontsize=9.5)
    ax.axvline(0, color=TEXT_COL, lw=1.3, linestyle="--", alpha=0.65)
    ax.set_xlabel("SHAP value  (contribution to log-odds of fraud)", fontsize=11)
    ax.set_title(
        f"SHAP Beeswarm — Top {top_n} Features  ({n_pts:,} samples)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_shap_waterfall(shap_vals: np.ndarray,
                         base_value: float,
                         X_test: pd.DataFrame,
                         y_test: pd.Series,
                         y_prob: np.ndarray,
                         feat_cols: list,
                         save_path: str,
                         top_n: int = 12):
    """
    Custom SHAP waterfall for the test sample with the highest fraud probability.
    Shows how the top-N features cumulatively shift the prediction from the
    base (expected) value to the final model output.

    Red bars  -> push toward fraud (positive SHAP)
    Teal bars -> push toward legitimate (negative SHAP)
    """
    # Identify highest-confidence fraud in test set
    fraud_rows   = np.where(y_test.values == 1)[0]
    fraud_probs  = y_prob[fraud_rows]
    best_local   = fraud_rows[np.argmax(fraud_probs)]
    sample_shap  = shap_vals[best_local]
    sample_feat  = X_test.iloc[best_local]
    sample_prob  = float(y_prob[best_local])

    # Select top-N features by |SHAP| for this sample
    order_desc   = np.argsort(np.abs(sample_shap))[::-1]
    top_idx      = order_desc[:top_n]
    rest_sum     = sample_shap[order_desc[top_n:]].sum()

    # Build list from largest -> smallest (reverse so largest is at top in barh)
    names  = [feat_cols[i] for i in top_idx][::-1]
    shaps  = sample_shap[top_idx][::-1]
    fvals  = [sample_feat.iloc[i] for i in top_idx][::-1]

    # Starting x for each bar = base_value + cumulative previous
    starts = [base_value + sample_shap[top_idx][::-1][:k].sum()
              for k in range(len(shaps))]

    n = len(shaps)
    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.68)), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    for i, (nm, sv, fv, st) in enumerate(zip(names, shaps, fvals, starts)):
        color = COLOR_REC if sv >= 0 else COLOR_PREC
        ax.barh(i, sv, left=st, color=color, edgecolor=CHARCOAL,
                height=0.62, alpha=0.85)
        # Label: feature name + value + SHAP contribution
        label_x = st + sv / 2.0
        ax.text(label_x, i, f"{nm}={fv:.3f}  ({sv:+.4f})",
                ha="center", va="center", fontsize=8.0,
                color="#FFFFFF", fontweight="bold")

    # Base and final prediction lines
    final_log = base_value + float(sample_shap.sum())
    ax.axvline(base_value, color=GRID_COL, lw=2.0, linestyle="--",
               label=f"Base value: {base_value:.4f}  (P={_sigmoid(base_value):.4f})")
    ax.axvline(final_log, color=COLOR_F2, lw=2.0, linestyle="-",
               label=f"Model output: {final_log:.4f}  (P={sample_prob:.4f})")

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("Log-odds  (SHAP contribution)", fontsize=11)
    ax.set_title(
        f"SHAP Waterfall — Highest-Confidence Fraud Prediction\n"
        f"Predicted P(fraud) = {sample_prob:.4f}  |  True label: FRAUD  |  "
        f"Top {top_n} features shown",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.legend(handles=[
        Patch(color=COLOR_REC,  label="Pushes toward FRAUD (positive SHAP)"),
        Patch(color=COLOR_PREC, label="Pushes toward LEGIT  (negative SHAP)"),
        Patch(color=GRID_COL,   label=f"Base: {base_value:.4f}"),
        Patch(color=COLOR_F2,   label=f"Output: {final_log:.4f}  P={sample_prob:.4f}"),
    ], fontsize=8.5, loc="lower right")
    ax.grid(axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_shap_dependence_top5(shap_vals: np.ndarray,
                               X_test: pd.DataFrame,
                               feat_cols: list,
                               top5_df: pd.DataFrame,
                               save_path: str):
    """
    One scatter panel per top-5 feature.
    X = feature value  |  Y = SHAP value  |  colour = feature value.
    The slope reveals how feature magnitude drives fraud risk.
    """
    np.random.seed(RANDOM_STATE)
    names = list(top5_df["Feature"])
    n     = len(names)

    fig, axes = plt.subplots(1, n, figsize=(n * 5.2, 5.2), facecolor=CHARCOAL)
    if n == 1:
        axes = [axes]

    for ax, fname in zip(axes, names):
        ax.set_facecolor(PANEL_BG)
        fi      = feat_cols.index(fname)
        fv      = X_test.iloc[:, fi].values.astype(float)
        sv      = shap_vals[:, fi]

        # Subsample for clarity
        idx  = np.random.choice(len(fv), min(SHAP_PLOT_SAMPLE, len(fv)),
                                replace=False)
        fv_s = fv[idx]
        sv_s = sv[idx]

        # Clip extreme feature values for visualisation
        lo, hi = np.percentile(fv_s, [2, 98])
        mask   = (fv_s >= lo) & (fv_s <= hi)

        sc = ax.scatter(fv_s[mask], sv_s[mask],
                        c=fv_s[mask], cmap="RdBu_r",
                        alpha=0.5, s=9, rasterized=True)
        ax.axhline(0, color=GRID_COL, lw=1.3, linestyle="--")
        ax.set_xlabel(fname, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(fname, fontsize=11, fontweight="bold")
        ax.grid(True)

        # Pearson r annotation
        corr = np.corrcoef(fv_s[mask], sv_s[mask])[0, 1]
        ax.text(0.04, 0.95, f"r={corr:+.3f}", transform=ax.transAxes,
                fontsize=9, color=TEXT_COL, va="top")

        plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)

    fig.suptitle("SHAP Dependence Plots — Top 5 Fraud Signal Features",
                 fontsize=13, fontweight="bold", color=TEXT_COL, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  STEP 4 — BUSINESS IMPACT
# =============================================================================

def compute_business_impact(df_thresh: pd.DataFrame,
                              y_test: pd.Series,
                              y_prob: np.ndarray,
                              cost_fn_usd: float,
                              annual_scale: float,
                              amount_series: pd.Series) -> pd.DataFrame:
    """
    Evaluate five business scenarios from catch-nothing to catch-everything,
    computing test-set and annualised costs plus detection rates.
    """
    cost_opt_thresh = df_thresh.loc[df_thresh["Total_Cost_USD"].idxmin(), "threshold"]
    f2_opt_thresh   = df_thresh.loc[df_thresh["F2"].idxmax(), "threshold"]

    scenarios = {
        "No Model  (flag nothing)": 0.9999,
        "Default   (0.50)":         0.50,
        "F2-Optimal (recall x2)":   f2_opt_thresh,
        "Cost-Optimal  *":          cost_opt_thresh,
        "Catch-All (flag all)":     0.0001,
    }

    rows = []
    for label, thresh in scenarios.items():
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        fp_cost   = fp * COST_FP_USD
        fn_mask   = (y_test.values == 1) & (y_pred == 0)
        fn_cost   = float(amount_series.values[fn_mask].sum()) * CHARGEBACK_MULTIPLIER
        tot_cost  = fp_cost + fn_cost
        ann_cost  = tot_cost * annual_scale / len(y_test)
        catch_pct = tp / max(tp + fn, 1) * 100
        falarm_pct= fp / max(fp + tn, 1) * 100

        rows.append({
            "Scenario":           label,
            "Threshold":          thresh,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Test_Cost_USD":      round(tot_cost, 2),
            "Annual_Cost_USD":    round(ann_cost, 0),
            "Fraud_Caught_pct":   round(catch_pct, 2),
            "False_Alarm_pct":    round(falarm_pct, 4),
        })

    return pd.DataFrame(rows)


def compute_cost_sensitivity(df_thresh: pd.DataFrame,
                               y_test: pd.Series,
                               annual_scale: float):
    """
    Sensitivity grid: for each (cost_fp, cost_fn) combination find the
    cost-optimal threshold and compute the resulting annual cost.
    Uses vectorised operations on the pre-computed FP/FN counts in df_thresh.
    Returns grid (n_fn x n_fp) in $M, plus the axis arrays.
    """
    cost_fp_vals = np.arange(10, 210, 20)     # $10 to $200 step $20  (10 values)
    cost_fn_vals = np.arange(50, 1050, 100)   # $50 to $1000 step $100 (10 values)

    fp_counts = df_thresh["FP"].values.astype(float)
    fn_counts = df_thresh["FN"].values.astype(float)

    grid = np.zeros((len(cost_fn_vals), len(cost_fp_vals)))
    for i, cfn in enumerate(cost_fn_vals):
        for j, cfp in enumerate(cost_fp_vals):
            costs      = fp_counts * cfp + fn_counts * cfn
            min_test   = costs.min()
            grid[i, j] = min_test * annual_scale / len(y_test) / 1e6  # -> $M

    return grid, cost_fp_vals, cost_fn_vals


# ── Business impact plots ────────────────────────────────────────────────────

def plot_business_impact_scenarios(df_impact: pd.DataFrame,
                                    annual_scale: float,
                                    save_path: str):
    """
    Dual-panel bar chart: (left) annual cost per scenario,
    (right) fraud detection rate per scenario.
    """
    labels  = df_impact["Scenario"].tolist()
    costs_m = df_impact["Annual_Cost_USD"].values / 1e6
    catch   = df_impact["Fraud_Caught_pct"].values

    bar_colors = ["#666688", "#7B68EE", COLOR_F2, COLOR_COST, "#666688"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=CHARCOAL)
    ax1.set_facecolor(PANEL_BG)
    ax2.set_facecolor(PANEL_BG)
    x = range(len(labels))

    # Left: Annual cost
    bars1 = ax1.bar(x, costs_m, color=bar_colors, edgecolor=CHARCOAL, alpha=0.85)
    for bar, val in zip(bars1, costs_m):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + costs_m.max() * 0.02,
                 f"${val:.2f}M", ha="center", va="bottom",
                 fontsize=9.5, color=TEXT_COL, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax1.set_ylabel("Estimated Annual Cost ($ million)", fontsize=11)
    ax1.set_title("Annual Business Cost by Strategy",
                  fontsize=12, fontweight="bold")
    ax1.grid(axis="y")

    # Right: Fraud caught %
    bars2 = ax2.bar(x, catch, color=bar_colors, edgecolor=CHARCOAL, alpha=0.85)
    for bar, val in zip(bars2, catch):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9.5, color=TEXT_COL, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax2.set_ylabel("Fraud Cases Caught (%)", fontsize=11)
    ax2.set_title("Fraud Detection Rate by Strategy",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 118)
    ax2.grid(axis="y")

    fig.suptitle(
        f"Business Impact Comparison  "
        f"(~{annual_scale/1e6:.1f}M transactions/year)  |  "
        f"*Cost-Optimal = recommended deployment threshold",
        fontsize=11, fontweight="bold", color=TEXT_COL, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_cost_sensitivity_heatmap(grid: np.ndarray,
                                   cost_fp_vals: np.ndarray,
                                   cost_fn_vals: np.ndarray,
                                   save_path: str):
    """
    Heatmap of minimum achievable annual cost ($M) at the cost-optimal threshold
    for each combination of FP and FN cost assumptions.
    Annotated with base-case assumption highlighted.
    """
    row_labels = [f"${v}" for v in cost_fn_vals]
    col_labels = [f"${v}" for v in cost_fp_vals]

    df_heat = pd.DataFrame(grid, index=row_labels, columns=col_labels)

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    mask = np.zeros_like(grid, dtype=bool)
    sns.heatmap(
        df_heat, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=ax, linewidths=0.4, linecolor=CHARCOAL,
        annot_kws={"size": 8.5},
        cbar_kws={"label": "Min Annual Cost ($M) at cost-optimal threshold"},
    )

    # Mark base-case cell
    base_fp_col = int(np.argmin(np.abs(cost_fp_vals - COST_FP_USD)))
    # base_fn_row is dynamic from avg fraud — just note in title
    ax.add_patch(plt.Rectangle((base_fp_col, 0), 1, len(cost_fn_vals),
                                fill=False, edgecolor=COLOR_F2, lw=2.5))
    ax.text(base_fp_col + 0.5, -0.6, "Base FP",
            color=COLOR_F2, ha="center", fontsize=8.5)

    ax.set_title(
        "Cost Sensitivity Analysis\n"
        "Annual Cost ($M) at Cost-Optimal Threshold — "
        "for Different FP/FN Cost Assumptions",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Cost per False Positive (USD) — Investigation & Friction", fontsize=10)
    ax.set_ylabel("Cost per False Negative (USD) — Fraud Loss + Chargebacks", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  STEP 5 — WRITE BUSINESS REPORT
# =============================================================================

def write_business_report(df_thresh: pd.DataFrame,
                           opt: dict,
                           top5_df: pd.DataFrame,
                           df_impact: pd.DataFrame,
                           annual_scale: float,
                           cost_fn_usd: float,
                           avg_fraud_amt: float,
                           n_test: int,
                           report_path: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as fh:
        def w(line: str = ""):
            fh.write(line + "\n")

        w("=" * 72)
        w("  CREDIT CARD FRAUD DETECTION — DAY 3 BUSINESS REPORT")
        w(f"  Generated    : {ts}")
        w(f"  Model        : XGBoost (best_model.pkl, Day 2)")
        w(f"  Test set     : {n_test:,} transactions")
        w("=" * 72)

        # Cost assumptions
        w()
        w("  COST ASSUMPTIONS")
        w("  " + "-" * 60)
        w(f"  Cost per False Positive (FP) : ${COST_FP_USD:.2f}")
        w(f"    Analyst investigation        : $15")
        w(f"    Customer service call        : $25")
        w(f"    Customer friction / churn    : $10")
        w(f"  Cost per False Negative (FN) : ${cost_fn_usd:.2f}")
        w(f"    Avg fraud transaction amt    : ${avg_fraud_amt:.2f}")
        w(f"    Chargeback multiplier        : {CHARGEBACK_MULTIPLIER}x")
        w(f"  Annual transaction volume    : ~{annual_scale:,.0f} txns/year")
        w(f"    Derived: dataset_rows / {DAYS_IN_DATASET:.0f} days x 365 days")

        # Threshold summary
        w()
        w("  THRESHOLD OPTIMISATION RESULTS")
        w("  " + "-" * 74)
        hdr = (f"  {'Criterion':<28} {'Thresh':>6}  "
               f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'F2':>6}  {'Cost':>10}")
        w(hdr)
        w("  " + "-" * 74)
        for name, row in opt.items():
            tag = "  << PRIMARY RECOMMENDATION" if "Cost" in name else ""
            w(f"  {name:<28} {row['threshold']:>6.2f}  "
              f"{row['Precision']:>6.4f}  {row['Recall']:>6.4f}  "
              f"{row['F1']:>6.4f}  {row['F2']:>6.4f}  "
              f"{row['Total_Cost_USD']:>10,.2f}{tag}")
        dft = df_thresh[df_thresh["threshold"].between(0.499, 0.501)].iloc[0]
        w("  " + "-" * 74)
        w(f"  {'Default (0.50)':<28} {dft['threshold']:>6.2f}  "
          f"{dft['Precision']:>6.4f}  {dft['Recall']:>6.4f}  "
          f"{dft['F1']:>6.4f}  {dft['F2']:>6.4f}  "
          f"{dft['Total_Cost_USD']:>10,.2f}  <- baseline")

        # SHAP top 5
        if not top5_df.empty:
            w()
            w("  TOP 5 FRAUD SIGNAL FEATURES  (SHAP Analysis)")
            w("  " + "-" * 72)
            w(f"  {'Rank':<4} {'Feature':<14} {'MeanAbsSHAP':>12} "
              f"{'MeanSHAP_Fraud':>14}  Direction")
            w("  " + "-" * 72)
            for rank, row in top5_df.iterrows():
                w(f"  {rank:<4} {row['Feature']:<14} "
                  f"{row['Mean_Abs_SHAP']:>12.6f} "
                  f"{row['Mean_SHAP_Fraud']:>14.6f}  "
                  f"{row['Direction']}")
            w()
            w("  Interpretation:")
            w("  SHAP values (log-odds scale). Positive = increases P(fraud).")
            w("  Mean_Abs_SHAP = average influence across all 56,746 test samples.")
            w("  Mean_SHAP_Fraud = average direction of influence on FRAUD samples.")

        # Business impact table
        w()
        w("  BUSINESS IMPACT COMPARISON  (annualised)")
        w("  " + "-" * 84)
        w(f"  {'Scenario':<32} {'Thresh':>7} {'Caught':>8} {'FalseAlm':>9} "
          f"{'AnnualCost':>14}")
        w("  " + "-" * 84)
        for _, row in df_impact.iterrows():
            w(f"  {row['Scenario']:<32} {row['Threshold']:>7.4f} "
              f"{row['Fraud_Caught_pct']:>7.1f}% "
              f"{row['False_Alarm_pct']:>8.4f}% "
              f"${row['Annual_Cost_USD']:>13,.0f}")

        # Savings analysis
        no_mdl = df_impact[df_impact["Scenario"].str.contains("No Model")
                           ]["Annual_Cost_USD"].values[0]
        dft_c  = df_impact[df_impact["Scenario"].str.contains("Default")
                           ]["Annual_Cost_USD"].values[0]
        opt_c  = df_impact[df_impact["Scenario"].str.contains("Cost-Optimal")
                           ]["Annual_Cost_USD"].values[0]

        w()
        w("  ANNUAL SAVINGS ANALYSIS")
        w("  " + "-" * 60)
        w(f"  No-model baseline cost     : ${no_mdl:>15,.0f} / year")
        w(f"  Default threshold (0.50)   : ${dft_c:>15,.0f} / year")
        w(f"  Cost-optimal threshold     : ${opt_c:>15,.0f} / year")
        w()
        sv1 = no_mdl - dft_c
        sv2 = no_mdl - opt_c
        sv3 = dft_c  - opt_c
        pct1 = sv1 / max(no_mdl, 1) * 100
        pct2 = sv2 / max(no_mdl, 1) * 100
        w(f"  Savings: No-model -> Default   : ${sv1:>12,.0f} / yr  ({pct1:.1f}%)")
        w(f"  Savings: No-model -> Optimal   : ${sv2:>12,.0f} / yr  ({pct2:.1f}%)")
        w(f"  Additional: Default -> Optimal : ${sv3:>12,.0f} / yr  (marginal gain)")

        # Recommendation
        cost_row = opt["Cost-Optimal (Business)"]
        w()
        w("  DEPLOYMENT RECOMMENDATION")
        w("  " + "-" * 60)
        w(f"  Recommended threshold : {cost_row['threshold']:.2f}")
        w(f"  Rationale: minimises combined FP investigation cost "
          f"(${COST_FP_USD:.0f}/FP) +")
        w(f"             FN chargeback loss (${cost_fn_usd:.0f}/FN).")
        w()
        w(f"  Expected performance at threshold {cost_row['threshold']:.2f}:")
        total_fraud = int(cost_row["TP"]) + int(cost_row["FN"])
        w(f"    Fraud caught      : {cost_row['TP']} / {total_fraud}  "
          f"({cost_row['Recall']*100:.1f}%)")
        w(f"    False alarms      : {cost_row['FP']} per {n_test:,} transactions  "
          f"({cost_row['FP']/n_test*100:.4f}%)")
        w(f"    Precision         : {cost_row['Precision']:.4f}")
        w(f"    F2 score          : {cost_row['F2']:.4f}")
        w(f"    Test-set cost     : ${cost_row['Total_Cost_USD']:,.2f}")
        w()
        w("  Monitor KPIs in production:")
        w("  - Weekly: Recall >= 75%  and  False-alarm rate <= 0.3%")
        w("  - Monthly: Recalibrate threshold if fraud pattern shifts > 10%")
        w("  - Quarterly: Retrain with new fraud labels + SHAP re-analysis")

        w()
        w("=" * 72)

    print(f"  Report saved -> {report_path}")


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main() -> dict:
    print("\n" + "#" * 68)
    print("#  FRAUD DETECTION — DAY 3")
    print("#  Threshold Tuning | SHAP Analysis | Business Impact")
    print(f"#  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 68)

    t_start = time.time()
    for d in [OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────
    (model, X_test, y_test, X_val, y_val, y_val_prob, X_train,
     y_prob, feat_cols,
     avg_fraud_amt, cost_fn_usd, annual_scale) = load_artifacts()

    # ── 2. Threshold Analysis ──────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  STEP 2 — THRESHOLD SWEEP  ({THRESH_MIN} to {THRESH_MAX}, "
          f"step {THRESH_STEP})")
    print("=" * 68)

    # Threshold sweep runs on VALIDATION data — test set is never seen here.
    df_thresh = compute_threshold_metrics(y_val, y_val_prob, cost_fn_usd, X_val["Amount"])
    opt       = find_optimal_thresholds(df_thresh)
    print_threshold_summary(df_thresh, opt, cost_fn_usd)

    # Freeze and persist the cost-optimal threshold found on validation
    best_threshold = float(opt["Cost-Optimal (Business)"]["threshold"])
    with open(BEST_THRESHOLD_PATH, "wb") as fh:
        pickle.dump(best_threshold, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Best threshold (val-tuned) : {best_threshold:.2f}")
    print(f"  Saved -> {BEST_THRESHOLD_PATH}")

    # Save validation threshold sweep CSV
    thresh_csv = os.path.join(RESULTS_DIR, "threshold_analysis_val.csv")
    df_thresh.to_csv(thresh_csv, index=False)
    print(f"  Saved -> {thresh_csv}  ({len(df_thresh)} rows)  [validation set]")

    # Plots — curves are on validation data (honest: same data used for selection)
    print("\n  Generating threshold plots …")
    plot_threshold_metrics(
        df_thresh, opt,
        os.path.join(PLOTS_DIR, "01_threshold_metrics.png"),
    )
    plot_threshold_cost(
        df_thresh, opt, cost_fn_usd,
        os.path.join(PLOTS_DIR, "02_threshold_cost_curve.png"),
    )
    # Confusion comparison: frozen val-tuned threshold applied ONCE to test set
    plot_confusion_comparison(
        y_test, y_prob,
        default_thresh=0.50,
        opt_thresh=best_threshold,
        save_path=os.path.join(PLOTS_DIR, "03_threshold_confusion_compare.png"),
    )

    # ── 3. SHAP Analysis ───────────────────────────────────────────────────
    shap_vals = None
    top5_df   = pd.DataFrame()

    if not _HAS_SHAP:
        print("\n  [SKIP] SHAP not installed — skipping SHAP step.")
    else:
        print("\n" + "=" * 68)
        print("  STEP 3 — SHAP ANALYSIS")
        print("=" * 68)

        shap_vals, base_value = compute_shap_values(model, X_test)
        top5_df = extract_top5_fraud_signals(
            shap_vals, X_test, y_test, feat_cols
        )

        # Save top-5 table
        shap_csv = os.path.join(RESULTS_DIR, "shap_top5_features.csv")
        top5_df.to_csv(shap_csv)
        print(f"\n  Saved -> {shap_csv}")

        # SHAP plots
        print("\n  Generating SHAP plots …")
        plot_shap_global_bar(
            shap_vals, feat_cols, TOP_N_SHAP_BAR,
            os.path.join(PLOTS_DIR, "04_shap_global_bar.png"),
        )
        plot_shap_beeswarm(
            shap_vals, X_test, feat_cols, TOP_N_BEESWARM,
            os.path.join(PLOTS_DIR, "05_shap_beeswarm.png"),
        )
        plot_shap_waterfall(
            shap_vals, base_value, X_test, y_test, y_prob, feat_cols,
            os.path.join(PLOTS_DIR, "06_shap_waterfall_fraud.png"),
        )
        plot_shap_dependence_top5(
            shap_vals, X_test, feat_cols, top5_df,
            os.path.join(PLOTS_DIR, "07_shap_dependence_top5.png"),
        )

    # ── 4. Business Impact ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 4 — BUSINESS IMPACT CALCULATION")
    print("=" * 68)

    df_impact = compute_business_impact(
        df_thresh, y_test, y_prob, cost_fn_usd, annual_scale, X_test["Amount"]
    )

    print(f"\n  Annual scale: ~{annual_scale:,.0f} transactions/year\n")
    print(f"  {'Scenario':<32} {'Thresh':>7} {'Caught':>8} {'AnnualCost':>15}")
    print("  " + "-" * 68)
    for _, row in df_impact.iterrows():
        print(f"  {row['Scenario']:<32} {row['Threshold']:>7.4f} "
              f"{row['Fraud_Caught_pct']:>7.1f}%  "
              f"${row['Annual_Cost_USD']:>13,.0f}")

    # Savings highlights
    no_mdl_cost  = df_impact[df_impact["Scenario"].str.contains("No Model")
                             ]["Annual_Cost_USD"].values[0]
    dft_cost     = df_impact[df_impact["Scenario"].str.contains("Default")
                             ]["Annual_Cost_USD"].values[0]
    opt_cost     = df_impact[df_impact["Scenario"].str.contains("Cost-Optimal")
                             ]["Annual_Cost_USD"].values[0]
    print(f"\n  Savings vs No-Model:   Default=${no_mdl_cost-dft_cost:>12,.0f}/yr")
    print(f"  Savings vs No-Model:   Optimal=${no_mdl_cost-opt_cost:>12,.0f}/yr")
    print(f"  Extra gain (Dft->Opt):         ${dft_cost-opt_cost:>12,.0f}/yr")

    print("\n  Generating business impact plots …")
    plot_business_impact_scenarios(
        df_impact, annual_scale,
        os.path.join(PLOTS_DIR, "08_business_impact_scenarios.png"),
    )
    grid, fp_ax, fn_ax = compute_cost_sensitivity(df_thresh, y_test, annual_scale)
    plot_cost_sensitivity_heatmap(
        grid, fp_ax, fn_ax,
        os.path.join(PLOTS_DIR, "09_cost_sensitivity_heatmap.png"),
    )

    # ── 5. Write report ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 5 — WRITING BUSINESS REPORT")
    print("=" * 68)

    write_business_report(
        df_thresh, opt, top5_df, df_impact,
        annual_scale, cost_fn_usd, avg_fraud_amt, len(y_test),
        os.path.join(RESULTS_DIR, "business_impact_report.txt"),
    )

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print("\n" + "#" * 68)
    print("#  DAY 3 COMPLETE")
    print("#" * 68)
    print(f"\n  Runtime : {elapsed:.1f}s")
    print(f"\n  Optimal Thresholds:")
    for name, row in opt.items():
        marker = "  <-- DEPLOY" if "Cost" in name else ""
        print(f"    {name:<28} : {row['threshold']:.2f}{marker}")

    if not top5_df.empty:
        print(f"\n  Top 5 Fraud Signals (SHAP):")
        for rank, row in top5_df.iterrows():
            print(f"    #{rank}  {row['Feature']:<14}  "
                  f"MeanAbsSHAP={row['Mean_Abs_SHAP']:.5f}  "
                  f"{'<--' if rank==1 else '   '}  "
                  f"{row['Direction']}")

    print(f"\n  Output artefacts:")
    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            kb    = os.path.getsize(fpath) / 1024
            print(f"    {fpath}  ({kb:,.1f} KB)")
    print()

    return {
        "df_thresh":    df_thresh,
        "opt":          opt,
        "shap_vals":    shap_vals,
        "top5_df":      top5_df,
        "df_impact":    df_impact,
        "annual_scale": annual_scale,
    }


if __name__ == "__main__":
    artefacts = main()
