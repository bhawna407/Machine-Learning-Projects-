# =============================================================================
#  CREDIT CARD FRAUD DETECTION — DAY 2
#  Model Training  |  Evaluation  |  Best-Model Selection
#
#  Inputs  (Day 1 outputs):
#    ../DAY 1/processed/train_raw.parquet   — 226,980 rows × 34 cols
#    ../DAY 1/processed/test_raw.parquet    —  56,746 rows × 34 cols
#
#  Outputs:
#    best_model.pkl                             <- winning model
#    output/plots/roc_curves.png
#    output/plots/pr_curves.png
#    output/plots/confusion_matrices.png
#    output/plots/feature_importance_random_forest.png
#    output/plots/feature_importance_xgboost.png
#    output/plots/feature_importance_lightgbm.png
#    output/results/model_comparison.csv
#    output/results/day2_results_report.txt
#    output/models/logistic_regression.pkl
#    output/models/random_forest.pkl
#    output/models/xgboost.pkl
#    output/models/lightgbm.pkl
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
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score,
    accuracy_score, confusion_matrix, classification_report,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed — XGBoost model will be skipped.")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    print("[WARN] lightgbm not installed — LightGBM model will be skipped.")


# =============================================================================
#  CONFIGURATION
# =============================================================================

RANDOM_STATE   = 42
TARGET_COL     = "Class"
TOP_N_FEATURES = 20

# Day 1 output paths (relative to the DAY 2 working directory)
TRAIN_PATH = os.path.join("..", "DAY 1", "processed", "train_raw.parquet")
TEST_PATH  = os.path.join("..", "DAY 1", "processed", "test_raw.parquet")

# Day 2 output directories
OUTPUT_DIR      = "output"
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR     = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR      = os.path.join(OUTPUT_DIR, "models")
BEST_MODEL_PATH     = "best_model.pkl"
BEST_THRESHOLD_PATH = "best_threshold.pkl"

# Validation set — carved from training data; used ONLY for threshold tuning in Day 3.
# Never touches the test set.
VAL_X_PATH = os.path.join(RESULTS_DIR, "validation_X.parquet")
VAL_Y_PATH = os.path.join(RESULTS_DIR, "validation_y.parquet")

# Colour palette — mirrors Day 1
CHARCOAL = "#1C1C2E"
PANEL_BG = "#252540"
GRID_COL = "#3A3A5C"
TEXT_COL = "#E0E0F0"

MODEL_COLORS = {
    "Logistic Regression": "#7B68EE",
    "Random Forest":       "#00D4AA",
    "XGBoost":             "#FF4C61",
    "LightGBM":            "#FFD700",
}

plt.rcParams.update({
    "figure.facecolor": CHARCOAL,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   GRID_COL,
    "axes.labelcolor":  TEXT_COL,
    "axes.titlecolor":  TEXT_COL,
    "xtick.color":      TEXT_COL,
    "ytick.color":      TEXT_COL,
    "text.color":       TEXT_COL,
    "grid.color":       GRID_COL,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COL,
    "font.family":      "DejaVu Sans",
})


# =============================================================================
#  STEP 1 — DATA LOADING
# =============================================================================

def load_data(train_path: str, test_path: str):
    """
    Load Day 1 parquet outputs.
    Separates features from target; returns X/y splits + feature column list.
    Raises AssertionError if expected files or columns are missing.
    """
    print("\n" + "=" * 68)
    print("  STEP 1 — LOAD DAY 1 OUTPUTS")
    print("=" * 68)

    assert os.path.exists(train_path), \
        f"Train file not found: {os.path.abspath(train_path)}"
    assert os.path.exists(test_path), \
        f"Test file not found: {os.path.abspath(test_path)}"

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    for name, df in [("train", train_df), ("test", test_df)]:
        assert TARGET_COL in df.columns, \
            f"Target column '{TARGET_COL}' missing from {name} data"

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    # Verify column parity before any processing
    assert list(train_df[feature_cols].columns) == list(test_df[feature_cols].columns), \
        "Train and test feature columns do not match"

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[TARGET_COL].copy()

    for label, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        n_fraud = int(y.sum())
        print(
            f"  {label:<6}: {len(X):>8,} rows | {len(feature_cols)} features | "
            f"fraud={n_fraud:>5,} ({y.mean()*100:.4f}%) | "
            f"legit={len(y)-n_fraud:>8,} ({(1-y.mean())*100:.4f}%)"
        )

    print(f"\n  Features: {', '.join(feature_cols[:6])} … {', '.join(feature_cols[-3:])}")
    print(f"  Total feature columns : {len(feature_cols)}")
    print("  Load: [OK]")

    return X_train, X_test, y_train, y_test, feature_cols


# =============================================================================
#  STEP 1b — VALIDATION SPLIT  (from training data only)
# =============================================================================

def create_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carve a stratified 80/20 validation set out of the training data.

    The validation set is used exclusively for threshold tuning in Day 3.
    SMOTE is applied only to the remaining 80% (X_train_main).
    The test set is never touched until final evaluation.

    Returns: X_train_main, X_val, y_train_main, y_val
    """
    print("\n" + "=" * 68)
    print("  STEP 1b — VALIDATION SPLIT  (from training data; for threshold tuning)")
    print("=" * 68)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size    = 0.20,
        stratify     = y_train,
        random_state = RANDOM_STATE,
    )
    X_tr  = X_tr.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_tr  = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    print(
        f"  Train-main : {len(X_tr):>8,} rows | "
        f"fraud={int(y_tr.sum()):>5,} ({y_tr.mean()*100:.4f}%)"
    )
    print(
        f"  Validation : {len(X_val):>8,} rows | "
        f"fraud={int(y_val.sum()):>5,} ({y_val.mean()*100:.4f}%)"
    )
    print("  Validation split: [OK]")
    return X_tr, X_val, y_tr, y_val


# =============================================================================
#  STEP 2 — SMOTE  (train only)
# =============================================================================

def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample minority class on training set only.
    k_neighbors is computed dynamically so SMOTE never requests
    more neighbours than available minority samples.
    """
    print("\n" + "=" * 68)
    print("  STEP 2 — SMOTE  (train only — test set is never touched)")
    print("=" * 68)

    minority_count = int(y_train.sum())
    k_neighbors    = min(5, minority_count - 1)

    print(
        f"  Before  : {len(X_train):>8,} rows | "
        f"fraud={minority_count:>5,} ({y_train.mean()*100:.4f}%) | "
        f"legit={int((y_train == 0).sum()):>8,}"
    )
    print(f"  k_neighbors = {k_neighbors}  (dynamic: min(5, minority_count - 1))")

    smote = SMOTE(
        sampling_strategy="minority",
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )
    X_np, y_np = smote.fit_resample(X_train, y_train)

    X_sm = pd.DataFrame(X_np, columns=X_train.columns)
    y_sm = pd.Series(y_np, name=TARGET_COL)

    n_fraud_sm = int(y_sm.sum())
    n_legit_sm = int((y_sm == 0).sum())
    print(
        f"  After   : {len(X_sm):>8,} rows | "
        f"fraud={n_fraud_sm:>5,} ({y_sm.mean()*100:.4f}%) | "
        f"legit={n_legit_sm:>8,} ({(1-y_sm.mean())*100:.4f}%)"
    )

    return X_sm, y_sm


# =============================================================================
#  SANITY CHECKS
# =============================================================================

def sanity_check_smote(X_train_raw: pd.DataFrame, y_train_raw: pd.Series,
                        X_sm: pd.DataFrame,       y_sm:        pd.Series):
    """
    Five-point verification that SMOTE behaved correctly.
    Raises AssertionError on any failure so the pipeline halts early.
    """
    print("\n  [SANITY] SMOTE checks …")

    # 1. Dataset must grow
    assert len(X_sm) > len(X_train_raw), \
        "SMOTE did not increase the dataset size"
    print(f"    [OK] Size grew: {len(X_train_raw):,} -> {len(X_sm):,} rows")

    # 2. Minority fraction must increase
    assert y_sm.mean() > y_train_raw.mean(), \
        "SMOTE did not increase the minority class ratio"
    print(
        f"    [OK] Minority fraction: "
        f"{y_train_raw.mean()*100:.4f}% -> {y_sm.mean()*100:.4f}%"
    )

    # 3. No NaN values introduced by SMOTE
    nan_total = X_sm.isnull().sum().sum()
    assert nan_total == 0, f"SMOTE introduced {nan_total} NaN values"
    print("    [OK] No NaN values in SMOTE output")

    # 4. Column names must be preserved exactly
    assert list(X_sm.columns) == list(X_train_raw.columns), \
        "Feature column names changed after SMOTE"
    print(f"    [OK] Feature columns preserved ({len(X_sm.columns)})")

    # 5. Both classes present in SMOTE output
    assert set(y_sm.unique()) == {0, 1}, \
        f"Expected classes {{0, 1}}, got {set(y_sm.unique())}"
    print("    [OK] Both classes present in SMOTE output")

    print("  [SANITY] SMOTE: ALL 5 CHECKS PASSED\n")


def sanity_check_leakage(X_train_sm: pd.DataFrame, X_test: pd.DataFrame,
                          y_test_snap: pd.Series,   y_test: pd.Series):
    """
    Verify the test set was not contaminated by the SMOTE step.
    Four checks: shape, label integrity, column parity, row-hash overlap.
    """
    print("  [SANITY] Leakage checks …")

    # 1. Test shape unchanged
    assert X_test.shape == (len(y_test), X_test.shape[1]), \
        "X_test shape inconsistent with y_test length"
    print(f"    [OK] Test set shape intact: {X_test.shape}")

    # 2. y_test labels not modified
    assert y_test_snap.equals(y_test), \
        "y_test was modified — possible data leakage"
    print("    [OK] y_test labels unmodified")

    # 3. Train and test share the same feature columns
    assert list(X_train_sm.columns) == list(X_test.columns), \
        "Train/test feature columns do not match"
    print(f"    [OK] Train/test feature columns match ({len(X_test.columns)} cols)")

    # 4. Hash-based row overlap — full train vs full test, index-independent
    train_hashes = set(
        pd.util.hash_pandas_object(X_train_sm, index=False).values
    )
    test_hashes = set(
        pd.util.hash_pandas_object(X_test, index=False).values
    )
    overlap = len(train_hashes & test_hashes)
    assert overlap == 0, (
        f"Potential leakage detected: {overlap} duplicate rows between "
        f"SMOTE-train and test sets"
    )
    print(f"    [OK] Row-hash overlap (full SMOTE-train vs full test): 0 (no leakage)")

    print("  [SANITY] Leakage: ALL 4 CHECKS PASSED\n")


def sanity_check_ranking(comparison_df: pd.DataFrame):
    """Verify model comparison table is correctly sorted and metric values are sane."""
    print("\n  [SANITY] Ranking checks …")

    ap_vals  = comparison_df["Avg Precision"].values
    auc_vals = comparison_df["ROC-AUC"].values

    # 1. Strict non-increasing Avg Precision (PR-AUC) order — primary sort for imbalanced fraud
    assert all(ap_vals[i] >= ap_vals[i + 1] for i in range(len(ap_vals) - 1)), \
        "Comparison table is not sorted by Avg Precision descending"
    print("    [OK] Models sorted by Avg Precision / PR-AUC (descending)")

    # 2. All Avg Precision and ROC-AUC values in valid range
    assert all(0.0 <= v <= 1.0 for v in ap_vals), \
        f"Avg Precision outside [0, 1]: {ap_vals}"
    assert all(0.0 <= v <= 1.0 for v in auc_vals), \
        f"ROC-AUC outside [0, 1]: {auc_vals}"
    print("    [OK] All Avg Precision and ROC-AUC values in [0, 1]")

    # 3. F1 and F2 scores non-negative
    f1_vals = comparison_df["F1"].values
    f2_vals = comparison_df["F2"].values
    assert all(v >= 0 for v in f1_vals), f"Negative F1 score(s): {f1_vals}"
    assert all(v >= 0 for v in f2_vals), f"Negative F2 score(s): {f2_vals}"
    print("    [OK] All F1 and F2 scores >= 0")

    # 4. Precision and Recall in [0, 1]
    for metric in ["Precision", "Recall"]:
        vals = comparison_df[metric].values
        assert all(0.0 <= v <= 1.0 for v in vals), \
            f"{metric} values outside [0, 1]: {vals}"
    print("    [OK] Precision and Recall in [0, 1]")

    # 5. Best model row is rank 1
    assert comparison_df.index[0] == 1, \
        "Rank-1 model is not at position 0 of the table"
    print(f"    [OK] Rank-1 model confirmed: {comparison_df.iloc[0]['Model']}")

    print("  [SANITY] Ranking: ALL 7 CHECKS PASSED\n")


# =============================================================================
#  STEP 3 — MODEL DEFINITIONS
# =============================================================================

def get_models() -> dict:
    """
    Return {name: unfitted estimator} for each classifier.

    Logistic Regression is wrapped in a StandardScaler Pipeline because
    the Amount-derived features still span a wide numerical range even after
    log-transformation; the V1–V28 PCA features are already centered but
    scaling doesn't hurt them.  Tree models are scale-invariant.
    """
    models: dict = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    return models


# =============================================================================
#  STEP 4 — TRAINING
# =============================================================================

def train_models(models: dict,
                 X_train: pd.DataFrame,
                 y_train: pd.Series) -> dict:
    """Fit every model on the SMOTE-balanced training set."""
    print("\n" + "=" * 68)
    print("  STEP 4 — TRAINING  (SMOTE-balanced train set)")
    print("=" * 68)
    print(f"  Training set : {len(X_train):,} rows | "
          f"fraud={int(y_train.sum()):,} | legit={int((y_train==0).sum()):,}")

    trained: dict = {}
    for name, model in models.items():
        print(f"\n  [{name}] training …", end="", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        trained[name] = model
        print(f" done in {elapsed:.1f}s")

    print(f"\n  Trained : {', '.join(trained)}")
    return trained


# =============================================================================
#  STEP 5 — EVALUATION
# =============================================================================

def _evaluate_single(name: str, model,
                     X_test: pd.DataFrame,
                     y_test: pd.Series) -> dict:
    """
    Full evaluation for one model.
    predict_proba() is used for all probability-based metrics
    (ROC-AUC, Avg Precision, ROC/PR curves).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        "name":          name,
        "ROC-AUC":       round(roc_auc_score(y_test, y_prob), 6),
        "Avg Precision": round(average_precision_score(y_test, y_prob), 6),
        "Precision":     round(precision_score(y_test, y_pred, zero_division=0), 6),
        "Recall":        round(recall_score(y_test, y_pred, zero_division=0), 6),
        "F1":            round(f1_score(y_test, y_pred, zero_division=0), 6),
        "F2":            round(fbeta_score(y_test, y_pred, beta=2, zero_division=0), 6),
        "Accuracy":      round(accuracy_score(y_test, y_pred), 6),
        "Confusion Matrix":     confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(
            y_test, y_pred,
            target_names=["Legit (0)", "Fraud (1)"],
            digits=4,
        ),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def evaluate_all(trained: dict,
                 X_test: pd.DataFrame,
                 y_test: pd.Series) -> dict:
    """Evaluate all trained models on the untouched test set."""
    print("\n" + "=" * 68)
    print("  STEP 5 — EVALUATION  (untouched test set)")
    print("=" * 68)

    results: dict = {}
    for name, model in trained.items():
        print(f"\n  [{name}]")
        res = _evaluate_single(name, model, X_test, y_test)
        results[name] = res
        print(f"    ROC-AUC   : {res['ROC-AUC']:.6f}")
        print(f"    Precision : {res['Precision']:.6f}")
        print(f"    Recall    : {res['Recall']:.6f}")
        print(f"    F1        : {res['F1']:.6f}")
        print(f"    Accuracy  : {res['Accuracy']:.6f}")
        tn, fp, fn, tp = res["Confusion Matrix"].ravel()
        print(f"    Confusion : TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Assemble a comparison DataFrame for fraud detection.
    Ranked by Avg Precision (PR-AUC) descending; F2 then Recall as tiebreakers.
    F2 (beta=2) weights recall twice as heavily as precision — correct for fraud.
    """
    rows = [
        {
            "Model":         res["name"],
            "ROC-AUC":       res["ROC-AUC"],
            "Avg Precision": res["Avg Precision"],
            "Precision":     res["Precision"],
            "Recall":        res["Recall"],
            "F1":            res["F1"],
            "F2":            res["F2"],
            "Accuracy":      res["Accuracy"],
        }
        for res in results.values()
    ]
    # Fraud datasets are highly imbalanced.
    # Models are ranked primarily by PR-AUC (Average Precision)
    # because it better reflects fraud-detection quality than
    # Accuracy on rare-event datasets.
    df = (
        pd.DataFrame(rows)
        .sort_values(["Avg Precision", "F2", "Recall"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    df.index = df.index + 1   # 1-based rank
    df.index.name = "Rank"
    return df


def print_comparison_table(comparison_df: pd.DataFrame):
    """Pretty-print the ranked model comparison table to stdout."""
    print("\n" + "=" * 68)
    print("  MODEL COMPARISON TABLE  (ranked by Avg Precision v, F2 v, Recall v)")
    print("=" * 68)
    cols = ["Model", "ROC-AUC", "Avg Precision", "Precision", "Recall", "F1", "F2", "Accuracy"]
    print(comparison_df[cols].to_string(float_format="{:.6f}".format))
    print()


# =============================================================================
#  STEP 6 — PLOTS
# =============================================================================

def plot_roc_curves(results: dict, y_test: pd.Series, save_path: str):
    """Single figure — ROC curves for every model + random baseline."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRID_COL,
            linewidth=1.5, label="Random Classifier  (AUC = 0.500)", zorder=1)

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        color = MODEL_COLORS.get(name, "#FFFFFF")
        ax.plot(fpr, tpr, linewidth=2.2, color=color,
                label=f"{name}  (AUC = {res['ROC-AUC']:.4f})", zorder=2)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right",
              framealpha=0.9, edgecolor=GRID_COL)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_pr_curves(results: dict, y_test: pd.Series, save_path: str):
    """Single figure — Precision-Recall curves for every model + no-skill baseline."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    no_skill = float(y_test.mean())
    ax.axhline(no_skill, linestyle="--", color=GRID_COL, linewidth=1.5,
               label=f"No-Skill Baseline  (AP = {no_skill:.4f})", zorder=1)

    for name, res in results.items():
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, res["y_prob"])
        color = MODEL_COLORS.get(name, "#FFFFFF")
        ax.plot(rec_arr, prec_arr, linewidth=2.2, color=color,
                label=f"{name}  (AP = {res['Avg Precision']:.4f})", zorder=2)

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper right",
              framealpha=0.9, edgecolor=GRID_COL)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_confusion_matrices(results: dict, save_path: str):
    """2-row grid of annotated confusion-matrix heatmaps."""
    n = len(results)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 6, n_rows * 5),
                             facecolor=CHARCOAL)
    axes = np.array(axes).flatten()

    for ax, (name, res) in zip(axes, results.items()):
        ax.set_facecolor(PANEL_BG)
        cm     = res["Confusion Matrix"]
        labels = ["Legit (0)", "Fraud (1)"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar=False, linewidths=0.5,
            annot_kws={"size": 13, "weight": "bold"},
        )
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{name}\n"
            f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}",
            fontsize=10, fontweight="bold", color=TEXT_COL,
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices — All Models",
                 fontsize=14, fontweight="bold", color=TEXT_COL, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def _extract_feature_importances(model,
                                  feature_names: list,
                                  model_name: str) -> pd.Series:
    """
    Pull feature importances out of a fitted estimator.
    Handles sklearn Pipeline wrappers transparently.
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        raise AttributeError(
            f"Cannot extract feature importances from {model_name} "
            f"(no feature_importances_ or coef_)."
        )

    series = pd.Series(importances, index=feature_names)
    return series.sort_values(ascending=False)


def plot_feature_importance(model, feature_names: list,
                             model_name: str, top_n: int,
                             save_path: str):
    """
    Horizontal bar chart of top-N feature importances for a single model.
    The most-important feature is always at the top.
    """
    imp = _extract_feature_importances(model, feature_names, model_name)
    imp = imp / imp.sum()                        # normalise: values represent % of total
    top = imp.head(top_n).sort_values()          # ascending -> barh puts best at top

    color = MODEL_COLORS.get(model_name, "#7B68EE")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)), facecolor=CHARCOAL)
    ax.set_facecolor(PANEL_BG)

    y_pos = range(len(top))
    bars  = ax.barh(y_pos, top.values, color=color,
                    edgecolor=CHARCOAL, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index, fontsize=10)

    # Annotate bar tips
    x_offset = top.values.max() * 0.008
    for bar, val in zip(bars, top.values):
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}",
            va="center", fontsize=8, color=TEXT_COL,
        )

    ax.set_xlabel("Normalised Importance (% of total)", fontsize=12)
    ax.set_title(
        f"Top-{top_n} Feature Importances — {model_name}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(right=top.values.max() * 1.15)
    ax.grid(axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_feature_importance_comparison(trained: dict,
                                        feature_names: list,
                                        top_n: int,
                                        save_path: str):
    """
    Side-by-side top-N comparison across all tree models in one figure.
    This supplements the individual plots.
    """
    tree_names = [n for n in ("Random Forest", "XGBoost", "LightGBM")
                  if n in trained]
    if not tree_names:
        return

    n_models = len(tree_names)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(n_models * 7, max(7, top_n * 0.4)),
                              facecolor=CHARCOAL)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, tree_names):
        ax.set_facecolor(PANEL_BG)
        imp = _extract_feature_importances(trained[name], feature_names, name)
        imp = imp / imp.sum()                    # normalise so models share a common scale
        top = imp.head(top_n).sort_values()
        color = MODEL_COLORS.get(name, "#FFFFFF")

        y_pos = range(len(top))
        ax.barh(y_pos, top.values, color=color, edgecolor=CHARCOAL, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalised Importance (% of total)", fontsize=10)
        ax.grid(axis="x")

    fig.suptitle(f"Top-{top_n} Feature Importance Comparison (Tree Models)",
                 fontsize=14, fontweight="bold", color=TEXT_COL, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  STEP 7 — MODEL SELECTION & SERIALISATION
# =============================================================================

def select_best_model(comparison_df: pd.DataFrame,
                       trained: dict) -> tuple:
    """
    Pick the model ranked #1 by Avg Precision (PR-AUC),
    with F2 and Recall used as secondary fraud-focused criteria.
    Returns (best_name: str, best_model: estimator).
    """
    print("\n" + "=" * 68)
    print("  STEP 7 — BEST MODEL SELECTION")
    print("=" * 68)
    print("  Selection criteria: primary = Avg Precision (PR-AUC) v,  Recall v,  ROC-AUC v")

    best_row   = comparison_df.iloc[0]
    best_name  = best_row["Model"]
    best_model = trained[best_name]

    print(f"\n  Winner         : {best_name}")
    print(f"  Avg Precision  : {best_row['Avg Precision']:.6f}  (PR-AUC — primary)")
    print(f"  F2             : {best_row['F2']:.6f}  (beta=2: recall weighted 2x precision)")
    print(f"  F1             : {best_row['F1']:.6f}")
    print(f"  Recall         : {best_row['Recall']:.6f}  (critical: catch every fraudster)")
    print(f"  ROC-AUC        : {best_row['ROC-AUC']:.6f}")

    # Runner-up info
    if len(comparison_df) > 1:
        r2 = comparison_df.iloc[1]
        print(f"\n  Runner-up  : {r2['Model']}  "
              f"(AP = {r2['Avg Precision']:.6f},  F2 = {r2['F2']:.6f},  "
              f"Recall = {r2['Recall']:.6f})")

    return best_name, best_model


def save_models(trained: dict, models_dir: str,
                best_model, best_model_path: str):
    """
    Persist every fitted model to models_dir as individual .pkl files,
    and the best model to best_model_path in the project root.
    """
    os.makedirs(models_dir, exist_ok=True)

    print("\n  Serialising all models …")
    for name, model in trained.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        path  = os.path.join(models_dir, fname)
        with open(path, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        kb = os.path.getsize(path) / 1024
        print(f"    {path:<55} ({kb:,.1f} KB)")

    with open(best_model_path, "wb") as fh:
        pickle.dump(best_model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    kb = os.path.getsize(best_model_path) / 1024
    print(f"\n  Best model -> {best_model_path}  ({kb:,.1f} KB)")


# =============================================================================
#  STEP 8 — REPORTING
# =============================================================================

def write_results_report(comparison_df: pd.DataFrame,
                          results: dict,
                          best_name: str,
                          report_path: str):
    """Write a complete plain-text report covering all metrics and best-model choice."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as fh:
        def w(line: str = ""):
            fh.write(line + "\n")

        w("=" * 70)
        w("  CREDIT CARD FRAUD DETECTION — DAY 2 RESULTS REPORT")
        w(f"  Generated  : {ts}")
        w(f"  Models     : {len(results)}")
        w(f"  Test rows  : {sum(len(r['y_pred']) for r in list(results.values())[:1]):,}")
        w("=" * 70)

        # --- Comparison table ---
        w()
        w("  MODEL COMPARISON TABLE  (ranked by Avg Precision v, F2 v, Recall v)")
        w("  " + "-" * 66)
        cols = ["Model", "ROC-AUC", "Avg Precision",
                "Precision", "Recall", "F1", "F2", "Accuracy"]
        w(comparison_df[cols].to_string(float_format="{:.6f}".format))

        # --- Best model ---
        w()
        w("  BEST MODEL")
        w("  " + "-" * 40)
        best_row = comparison_df[comparison_df["Model"] == best_name].iloc[0]
        w(f"  Name       : {best_name}")
        w(f"  Avg Precision : {best_row['Avg Precision']:.6f}  (PR-AUC — primary rank)")
        w(f"  F2            : {best_row['F2']:.6f}  (beta=2)")
        w(f"  ROC-AUC       : {best_row['ROC-AUC']:.6f}")
        w(f"  F1            : {best_row['F1']:.6f}")
        w(f"  Precision     : {best_row['Precision']:.6f}")
        w(f"  Recall        : {best_row['Recall']:.6f}")
        w(f"  Accuracy      : {best_row['Accuracy']:.6f}")
        w(f"  Serialised : best_model.pkl")

        # --- Per-model detail ---
        w()
        w("  DETAILED PER-MODEL RESULTS")
        w("  " + "-" * 66)
        for name, res in results.items():
            w()
            w(f"  ── {name} ──")
            for metric in ["ROC-AUC", "Avg Precision", "Precision",
                           "Recall", "F1", "F2", "Accuracy"]:
                w(f"    {metric:<16} : {res[metric]:.6f}")
            w()
            w("  Confusion Matrix (rows=Actual, cols=Predicted):")
            for row in res["Confusion Matrix"]:
                w("    " + "  ".join(f"{v:>8,}" for v in row))
            w()
            w("  Classification Report:")
            for line in res["Classification Report"].splitlines():
                w("    " + line)
            w()
            w("  " + "-" * 66)

        w()
        w("  OUTPUT ARTEFACTS")
        w("  " + "-" * 50)
        for root, _dirs, files in os.walk(OUTPUT_DIR):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                kb    = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
                w(f"  {fpath:<55}  {kb:>8,.1f} KB")
        w(f"  {'best_model.pkl':<55}  "
          f"{os.path.getsize(BEST_MODEL_PATH)/1024:>8,.1f} KB")

        w()
        w("=" * 70)

    print(f"  Report saved -> {report_path}")


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main() -> dict:
    print("\n" + "#" * 68)
    print("#  FRAUD DETECTION — DAY 2: MODEL TRAINING & EVALUATION")
    print(f"#  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Seed      : {RANDOM_STATE}")
    print("#" * 68)

    t_pipeline_start = time.time()

    # Create output directories up front
    for d in [OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Day 1 outputs
    X_train, X_test, y_train, y_test, feature_cols = load_data(
        TRAIN_PATH, TEST_PATH
    )
    y_test_snapshot = y_test.copy()   # kept for leakage verification

    # Validation data is reserved for threshold optimisation in Day 3.
    # It is separated before SMOTE to prevent synthetic samples from
    # influencing threshold-selection decisions.
    # ------------------------------------------------------------------
    # 1b. Carve validation set from training data (threshold tuning only)
    #     Must happen BEFORE SMOTE so val rows are never synthetically augmented.
    X_train_main, X_val, y_train_main, y_val = create_validation_split(
        X_train, y_train
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_val.to_parquet(VAL_X_PATH)
    y_val.to_frame().to_parquet(VAL_Y_PATH)
    print(f"\n  Validation parquets saved:")
    print(f"    {VAL_X_PATH}")
    print(f"    {VAL_Y_PATH}")

    # ------------------------------------------------------------------
    # Fraud transactions are extremely rare.
    # SMOTE is applied ONLY to the training data so that the model
    # sees more fraud examples during learning while keeping
    # validation and test sets representative of real-world data.
    # 2. Apply SMOTE to X_train_main only — validation set never enters SMOTE
    X_train_sm, y_train_sm = apply_smote(X_train_main, y_train_main)
    sanity_check_smote(X_train_main, y_train_main, X_train_sm, y_train_sm)
    sanity_check_leakage(X_train_sm, X_test, y_test_snapshot, y_test)

    # ------------------------------------------------------------------
    # 3. Define model zoo
    models = get_models()
    print(f"\n  Models to train: {', '.join(models)}")

    # ------------------------------------------------------------------
    # 4. Train on SMOTE-balanced data
    trained = train_models(models, X_train_sm, y_train_sm)

    # ------------------------------------------------------------------
    # 5. Evaluate on pristine test set
    results       = evaluate_all(trained, X_test, y_test)
    comparison_df = build_comparison_table(results)
    sanity_check_ranking(comparison_df)
    print_comparison_table(comparison_df)

    # ------------------------------------------------------------------
    # 6. Plots
    print("\n" + "=" * 68)
    print("  STEP 6 — GENERATING PLOTS")
    print("=" * 68)

    plot_roc_curves(
        results, y_test,
        os.path.join(PLOTS_DIR, "roc_curves.png"),
    )
    plot_pr_curves(
        results, y_test,
        os.path.join(PLOTS_DIR, "pr_curves.png"),
    )
    plot_confusion_matrices(
        results,
        os.path.join(PLOTS_DIR, "confusion_matrices.png"),
    )

    # Feature importance — one plot per tree model
    tree_model_names = [n for n in ("Random Forest", "XGBoost", "LightGBM")
                        if n in trained]
    for name in tree_model_names:
        slug = name.lower().replace(" ", "_")
        plot_feature_importance(
            trained[name], feature_cols, name,
            TOP_N_FEATURES,
            os.path.join(PLOTS_DIR, f"feature_importance_{slug}.png"),
        )

    # Combined comparison panel (all tree models side-by-side)
    plot_feature_importance_comparison(
        trained, feature_cols, TOP_N_FEATURES,
        os.path.join(PLOTS_DIR, "feature_importance_comparison.png"),
    )

    # ------------------------------------------------------------------
    # 7. Select & save best model
    best_name, best_model = select_best_model(comparison_df, trained)
    save_models(trained, MODELS_DIR, best_model, BEST_MODEL_PATH)

    # ------------------------------------------------------------------
    # 8. Persist results
    print("\n" + "=" * 68)
    print("  STEP 8 — SAVING RESULTS")
    print("=" * 68)

    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path)
    print(f"  Saved -> {csv_path}")

    write_results_report(
        comparison_df, results, best_name,
        os.path.join(RESULTS_DIR, "day2_results_report.txt"),
    )

    elapsed = time.time() - t_pipeline_start

    # ------------------------------------------------------------------
    # Final summary
    print("\n" + "#" * 68)
    print("#  DAY 2 PIPELINE COMPLETE")
    print("#" * 68)
    print(f"\n  Best model  : {best_name}")
    print(f"  ROC-AUC     : {comparison_df.iloc[0]['ROC-AUC']:.6f}")
    print(f"  F1          : {comparison_df.iloc[0]['F1']:.6f}")
    print(f"  Precision   : {comparison_df.iloc[0]['Precision']:.6f}")
    print(f"  Recall      : {comparison_df.iloc[0]['Recall']:.6f}")
    print(f"  Pipeline runtime : {elapsed:.1f}s")
    print(f"\n  Output artefacts:")
    print(f"    {BEST_MODEL_PATH}")
    for root, _dirs, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            kb    = os.path.getsize(fpath) / 1024
            print(f"    {fpath}  ({kb:,.1f} KB)")
    print()

    return {
        "best_name":     best_name,
        "best_model":    best_model,
        "comparison_df": comparison_df,
        "results":       results,
        "trained":       trained,
        "feature_cols":  feature_cols,
    }


if __name__ == "__main__":
    artefacts = main()
