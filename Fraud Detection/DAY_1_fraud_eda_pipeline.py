# =============================================================================
#  CREDIT CARD FRAUD DETECTION — EDA, FEATURE ENGINEERING & PREPROCESSING
#  Author  : Senior Data Scientist Pipeline
#  Dataset : creditcard.csv  (284,807 transactions x 31 features)
#  Python  : 3.8+
#  Libs    : pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves PNGs without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)

# imbalanced-learn (pip install imbalanced-learn)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Plot theme ────────────────────────────────────────────────────────────────
CHARCOAL   = "#1C1C2E"
PANEL_BG   = "#252540"
ACCENT_FR  = "#FF4C61"   # fraud  -> warm red
ACCENT_LG  = "#00D4AA"   # legit  -> teal
GRID_COLOR = "#3A3A5C"
TEXT_COLOR = "#E0E0F0"

plt.rcParams.update({
    "figure.facecolor"  : CHARCOAL,
    "axes.facecolor"    : PANEL_BG,
    "axes.edgecolor"    : GRID_COLOR,
    "axes.labelcolor"   : TEXT_COLOR,
    "axes.titlecolor"   : TEXT_COLOR,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "text.color"        : TEXT_COLOR,
    "grid.color"        : GRID_COLOR,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.4,
    "font.family"       : "DejaVu Sans",
    "legend.facecolor"  : PANEL_BG,
    "legend.edgecolor"  : GRID_COLOR,
})

EXPECTED_FEATURES = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]


# =============================================================================
# SECTION 1 -- DATA INTEGRITY & VERIFICATION
# =============================================================================

def load_and_verify(filepath: str) -> pd.DataFrame:
    """Load CSV, run integrity checks, and return the clean dataframe."""

    print("=" * 70)
    print("  SECTION 1 - DATA INTEGRITY & VERIFICATION")
    print("=" * 70)

    df = pd.read_csv(filepath)

    # Shape
    print(f"\n[INFO] Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Feature presence
    missing_cols  = [c for c in EXPECTED_FEATURES if c not in df.columns]
    extra_cols    = [c for c in df.columns if c not in EXPECTED_FEATURES]
    all_present   = len(missing_cols) == 0

    print(f"[INFO] Expected 31 features  : {'[OK] All present' if all_present else '[FAIL] MISSING: ' + str(missing_cols)}")
    if extra_cols:
        print(f"[WARN] Unexpected columns    : {extra_cols}")

    # Null values
    null_counts  = df.isnull().sum()
    total_nulls  = null_counts.sum()
    print(f"\n[INFO] Null value check      : {'[OK] No nulls found' if total_nulls == 0 else '[FAIL] Nulls detected!'}")
    if total_nulls > 0:
        print(null_counts[null_counts > 0].to_string())

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    print(f"[INFO] Duplicate rows        : {n_dupes:,}  {'<- removing...' if n_dupes else '[OK]'}")
    if n_dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Shape after dedup     : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Dtypes summary
    print(f"\n[INFO] Data types:\n{df.dtypes.value_counts().to_string()}")
    print(f"\n[INFO] Class value counts:\n{df['Class'].value_counts().to_string()}")

    return df


# =============================================================================
# SECTION 2 -- EDA & VISUALIZATION
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """Produce a four-panel EDA dashboard and print imbalance statistics."""

    print("\n" + "=" * 70)
    print("  SECTION 2 - EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    print("=" * 70)

    # Class imbalance stats
    class_counts = df["Class"].value_counts()
    class_pct    = df["Class"].value_counts(normalize=True) * 100

    print("\n[INFO] Class distribution:")
    print(f"  Legitimate (0) : {class_counts[0]:>8,}  ({class_pct[0]:.4f}%)")
    print(f"  Fraudulent  (1) : {class_counts[1]:>8,}  ({class_pct[1]:.4f}%)")
    print(f"  Imbalance ratio : 1 fraud per {class_counts[0]//class_counts[1]:,} legitimate transactions")

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    # Dashboard — 2 x 2 grid
    fig = plt.figure(figsize=(18, 14), facecolor=CHARCOAL)
    fig.suptitle(
        "Credit Card Fraud Detection — EDA Dashboard",
        fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel A: Amount distribution
    ax_a = fig.add_subplot(gs[0, 0])
    sns.histplot(
        df["Amount"], bins=120, color=ACCENT_LG, alpha=0.85,
        edgecolor="none", ax=ax_a, kde=True,
        line_kws={"linewidth": 2, "color": "#FFFFFF"},
    )
    ax_a.set_title("Transaction Amount Distribution", fontsize=13, fontweight="bold", pad=10)
    ax_a.set_xlabel("Amount (USD)")
    ax_a.set_ylabel("Count")
    ax_a.set_xlim(left=0)
    ax_a.grid(True)
    ax_a.axvline(df["Amount"].median(), color=ACCENT_FR, linestyle="--", linewidth=1.4,
                 label=f"Median ${df['Amount'].median():.2f}")
    ax_a.axvline(df["Amount"].mean(),   color="#FFD700",  linestyle=":",  linewidth=1.4,
                 label=f"Mean   ${df['Amount'].mean():.2f}")
    ax_a.legend(fontsize=9)

    # Panel B: Time distribution
    ax_b = fig.add_subplot(gs[0, 1])
    hours = (df["Time"] % 86400) / 3600
    sns.histplot(hours, bins=48, color="#7B68EE", alpha=0.85,
                 edgecolor="none", ax=ax_b, kde=True,
                 line_kws={"linewidth": 2, "color": "#FFFFFF"})
    ax_b.set_title("Transaction Time Distribution (Hour of Day)", fontsize=13, fontweight="bold", pad=10)
    ax_b.set_xlabel("Hour of Day (0-24)")
    ax_b.set_ylabel("Count")
    ax_b.set_xlim(0, 24)
    ax_b.set_xticks(range(0, 25, 4))
    ax_b.grid(True)

    # Panel C: Class imbalance bar + avg amount
    ax_c = fig.add_subplot(gs[1, 0])

    labels       = ["Legitimate (0)", "Fraudulent (1)"]
    counts       = [class_counts[0], class_counts[1]]
    bar_colors   = [ACCENT_LG, ACCENT_FR]
    bars = ax_c.bar(labels, counts, color=bar_colors, edgecolor=CHARCOAL,
                    linewidth=1.2, width=0.5, zorder=3)

    for bar, cnt, pct in zip(bars, counts, [class_pct[0], class_pct[1]]):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{cnt:,}\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=10, color=TEXT_COLOR, fontweight="bold",
        )

    ax_c.set_title("Class Distribution (Fraud vs. Legitimate)", fontsize=13, fontweight="bold", pad=10)
    ax_c.set_ylabel("Transaction Count")
    ax_c.grid(axis="y", zorder=0)
    ax_c.set_yscale("log")
    ax_c.set_ylim(1, counts[0] * 3)

    # Right y-axis: average amount overlay
    ax_c2 = ax_c.twinx()
    avg_amounts = [legit["Amount"].mean(), fraud["Amount"].mean()]
    ax_c2.plot(labels, avg_amounts, marker="D", color="#FFD700",
               linewidth=2.2, markersize=9, zorder=5, label="Avg Amount")
    for x_pos, avg in enumerate(avg_amounts):
        ax_c2.text(x_pos, avg + 4, f"${avg:.2f}", ha="center",
                   fontsize=9, color="#FFD700", fontweight="bold")
    ax_c2.set_ylabel("Average Amount (USD)", color="#FFD700")
    ax_c2.tick_params(axis="y", labelcolor="#FFD700")
    ax_c2.legend(loc="upper right", fontsize=9)

    # Panel D: Temporal density — fraud vs. legit
    ax_d = fig.add_subplot(gs[1, 1])

    fraud_hours = (fraud["Time"] % 86400) / 3600
    legit_hours = (legit["Time"] % 86400) / 3600

    sns.kdeplot(legit_hours, ax=ax_d, color=ACCENT_LG, linewidth=2.2,
                fill=True, alpha=0.25, label="Legitimate")
    sns.kdeplot(fraud_hours, ax=ax_d, color=ACCENT_FR, linewidth=2.2,
                fill=True, alpha=0.35, label="Fraudulent")

    ax_d.set_title("Transaction Density by Hour — Fraud vs. Legitimate", fontsize=13, fontweight="bold", pad=10)
    ax_d.set_xlabel("Hour of Day (0-24)")
    ax_d.set_ylabel("Density")
    ax_d.set_xlim(0, 24)
    ax_d.set_xticks(range(0, 25, 4))
    ax_d.legend(fontsize=10)
    ax_d.grid(True)

    plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=CHARCOAL)
    print("\n[INFO] EDA dashboard saved -> eda_dashboard.png")
    plt.close()    # FIX 3: close instead of show — Agg backend has no display


# =============================================================================
# SECTION 3 -- FEATURE ENGINEERING
# =============================================================================

def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Compute all three engineered features.
    FIX 1 + FIX 2: accepts split sets so engineering happens AFTER the split.
                   mean / std for amount_zscore are fitted on X_train only —
                   the exact same values are then applied to X_test, so no
                   test-set statistics ever contaminate the training pipeline.
    FIX 5: sigma clamped to 1e-8 to prevent ZeroDivisionError.
    Returns (X_train_eng, X_test_eng, train_stats).
    """

    print("\n" + "=" * 70)
    print("  SECTION 3 - FEATURE ENGINEERING")
    print("=" * 70)

    X_train = X_train.copy()
    X_test  = X_test.copy()

    # Feature 1 — deterministic, no fit needed
    X_train["hour_of_day"] = (X_train["Time"] % 86400) / 3600
    X_test["hour_of_day"]  = (X_test["Time"]  % 86400) / 3600

    # Feature 2 — deterministic, no fit needed
    X_train["amount_log"]  = np.log1p(X_train["Amount"])
    X_test["amount_log"]   = np.log1p(X_test["Amount"])

    # Feature 3 — fit on train only (FIX 1), guard divide-by-zero (FIX 5)
    train_mean = X_train["Amount"].mean()
    train_std  = X_train["Amount"].std(ddof=0)
    train_std  = max(train_std, 1e-8)          # FIX 5: zero-std guard
    X_train["amount_zscore"] = (X_train["Amount"] - train_mean) / train_std
    X_test["amount_zscore"]  = (X_test["Amount"]  - train_mean) / train_std

    train_stats = {"mean": train_mean, "std": train_std}

    print("\n[INFO] 3 new features added (train fit / test transform):")
    print("  * hour_of_day   - transaction hour (0-24 float)  [deterministic]")
    print("  * amount_log    - log(Amount + 1)                [deterministic]")
    print("  * amount_zscore - z-score of Amount              [train stats only]")
    print(f"\n  Train Amount mean : {train_mean:.6f}")
    print(f"  Train Amount std  : {train_std:.6f}  (used to scale test set)")

    print("\n[INFO] Train set feature stats:")
    new_feats = X_train[["hour_of_day", "amount_log", "amount_zscore"]].describe().round(4)
    print(new_feats.to_string())

    print("\n[INFO] Test set feature stats (scaled with train stats):")
    test_feats = X_test[["hour_of_day", "amount_log", "amount_zscore"]].describe().round(4)
    print(test_feats.to_string())

    # Sanity-check plots — distributions from the training set
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), facecolor=CHARCOAL)
    fig.suptitle("Engineered Features — Train Set Distributions", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR, y=1.02)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)

    palette = [ACCENT_LG, "#7B68EE", ACCENT_FR]
    for ax, feat, color in zip(axes, ["hour_of_day", "amount_log", "amount_zscore"], palette):
        sns.histplot(X_train[feat], bins=80, ax=ax, color=color, alpha=0.85,
                     edgecolor="none", kde=True,
                     line_kws={"linewidth": 2, "color": "#FFFFFF"})
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("engineered_features.png", dpi=150, bbox_inches="tight",
                facecolor=CHARCOAL)
    print("\n[INFO] Feature distribution plot saved -> engineered_features.png")
    plt.close()    # FIX 3: close instead of show

    return X_train, X_test, train_stats


# =============================================================================
# SECTION 4 -- TRAIN / TEST SPLIT
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Stratified 80/20 train-test split on the raw (pre-engineering) dataframe.
    FIX 7: reset_index(drop=True) on all four outputs ensures a clean 0-based
            integer index and prevents iloc/loc misalignment bugs downstream.
    """

    print("\n" + "=" * 70)
    print("  SECTION 4 - TRAIN / TEST SPLIT")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        random_state = RANDOM_STATE,
        stratify     = y,           # preserves class ratio in both splits
    )

    # FIX 7: reset indexes after split
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    print(f"\n[INFO] Full dataset  : {len(df):>8,} rows")
    print(f"[INFO] X_train shape : {X_train.shape}  (raw — engineering applied next)")
    print(f"[INFO] X_test  shape : {X_test.shape}")
    print(f"\n[INFO] Class balance - Training set:")
    train_dist = y_train.value_counts()
    train_pct  = y_train.value_counts(normalize=True) * 100
    print(f"  Legitimate (0) : {train_dist[0]:>7,}  ({train_pct[0]:.4f}%)")
    print(f"  Fraudulent  (1) : {train_dist[1]:>7,}  ({train_pct[1]:.4f}%)")

    print(f"\n[INFO] Class balance - Test set:")
    test_dist = y_test.value_counts()
    test_pct  = y_test.value_counts(normalize=True) * 100
    print(f"  Legitimate (0) : {test_dist[0]:>7,}  ({test_pct[0]:.4f}%)")
    print(f"  Fraudulent  (1) : {test_dist[1]:>7,}  ({test_pct[1]:.4f}%)")

    print("\n[INFO] Stratification verified — imbalance ratio preserved [OK]")
    print("[INFO] Index reset    — clean 0-based index on all four outputs [OK]")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 5 -- CLASS IMBALANCE HANDLING
# =============================================================================

def handle_imbalance(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
    """
    Approach A — SMOTE on StandardScaler-transformed training data.
    Approach B — Cost-sensitive classifier configuration.
    Approach C — imblearn Pipeline (scaler + SMOTE + RF) for CV-safe evaluation.
    FIX: StandardScaler fitted on X_train, applied to X_test (no leakage).
    FIX: sampling_strategy=0.2 (20% minority ratio) — ~3x less memory than 'minority' (1:1).
    FIX 6: k_neighbors derived dynamically as min(5, minority_count - 1).
    Returns: X_train_smote, y_train_smote, rf_balanced, xgb_params,
             scaler, X_train_scaled, X_test_scaled, rf_pipeline
    """

    print("\n" + "=" * 70)
    print("  SECTION 5 - CLASS IMBALANCE HANDLING")
    print("=" * 70)

    # Approach A: SMOTE
    print("\n-- Approach A: SMOTE (Synthetic Minority Over-Sampling) --------------")
    print("[INFO] Applying SMOTE to training data ...")

    # FIX: StandardScaler — fit on X_train only, transform X_test (no data leakage)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"[INFO] StandardScaler fitted on X_train  {X_train_scaled.shape}  [OK]")
    print(f"[INFO] StandardScaler applied to X_test  {X_test_scaled.shape}   [OK]")

    # FIX 6: dynamic k — prevents crash when minority_count < 6
    minority_count = int(y_train.sum())
    k              = min(5, minority_count - 1)
    print(f"[INFO] minority_count={minority_count}  k_neighbors={k}  (min(5, count-1))")

    # NOTE:
    # SMOTE on PCA-transformed financial data may create unrealistic synthetic samples.
    # In production, cost-sensitive learning is often preferred.
    smote = SMOTE(
        sampling_strategy = 0.2,          # FIX: was "minority" (1:1); 0.2 = 20% ratio — ~3x less memory
        k_neighbors       = k,
        random_state      = RANDOM_STATE,
    )

    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"[INFO] Before SMOTE - shape  : {X_train.shape}  |  fraud: {y_train.sum():,}")
    print(f"[INFO] After  SMOTE - shape  : {X_train_smote.shape}  |  fraud: {int(y_train_smote.sum()):,}")
    print(f"[INFO] Class balance after SMOTE:")
    smote_counts = pd.Series(y_train_smote).value_counts()
    print(f"  Legitimate (0) : {smote_counts[0]:>8,}")
    print(f"  Fraudulent  (1) : {smote_counts[1]:>8,}")

    # quick SMOTE visual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=CHARCOAL)
    fig.suptitle("Class Balance: Before vs. After SMOTE", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)

    before_counts = [y_train.value_counts()[0], y_train.value_counts()[1]]
    after_counts  = [smote_counts[0], smote_counts[1]]
    x_labels      = ["Legitimate (0)", "Fraudulent (1)"]
    colors        = [ACCENT_LG, ACCENT_FR]

    axes[0].bar(x_labels, before_counts, color=colors, edgecolor=CHARCOAL, width=0.5)
    axes[0].set_title("Before SMOTE", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y")
    for i, v in enumerate(before_counts):
        axes[0].text(i, v * 1.1, f"{v:,}", ha="center", fontsize=10,
                     color=TEXT_COLOR, fontweight="bold")

    axes[1].bar(x_labels, after_counts, color=colors, edgecolor=CHARCOAL, width=0.5)
    axes[1].set_title("After SMOTE (Balanced)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y")
    for i, v in enumerate(after_counts):
        axes[1].text(i, v * 1.005, f"{v:,}", ha="center", fontsize=10,
                     color=TEXT_COLOR, fontweight="bold")

    plt.tight_layout()
    plt.savefig("smote_balance.png", dpi=150, bbox_inches="tight", facecolor=CHARCOAL)
    print("[INFO] SMOTE balance plot saved -> smote_balance.png")
    plt.close()    # FIX 3: close instead of show

    # Approach B: Cost-Sensitive Learning
    print("\n-- Approach B: Cost-Sensitive / Class-Weight Configuration -----------")

    # Random Forest with class_weight='balanced'
    # 'balanced' auto-computes: w_i = n_samples / (n_classes * n_samples_i)
    rf_balanced = RandomForestClassifier(
        n_estimators  = 300,
        max_depth     = None,
        class_weight  = "balanced",       # <- key parameter
        n_jobs        = -1,
        random_state  = RANDOM_STATE,
    )
    print("[INFO] RandomForestClassifier configured with class_weight='balanced'")
    print(f"       Effective fraud weight ~= "
          f"{len(y_train) / (2 * y_train.sum()):.1f}x  "
          f"(n_samples / (n_classes x n_minority))")

    # XGBoost equivalent: scale_pos_weight = count(negative) / count(positive)
    neg_count        = int((y_train == 0).sum())
    pos_count        = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    print(f"\n[INFO] XGBoost cost-sensitive configuration:")
    print(f"       neg_count        : {neg_count:,}")
    print(f"       pos_count        : {pos_count:,}")
    print(f"       scale_pos_weight : {scale_pos_weight:.2f}")

    xgb_balanced_params = {
        "n_estimators"     : 300,
        "max_depth"        : 6,
        "learning_rate"    : 0.05,
        "scale_pos_weight" : scale_pos_weight,    # <- key parameter
        "eval_metric"      : "aucpr",             # area under PR curve — better for imbalance
        "use_label_encoder": False,
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
    }
    print(f"\n[INFO] XGBoost params dict ready (use with xgb.XGBClassifier(**params)):")
    for k_name, v in xgb_balanced_params.items():
        print(f"       {k_name:<22} : {v}")

    print("\n[SUMMARY] Imbalance strategy comparison:")
    print("  Approach A - SMOTE      : Creates synthetic fraud samples. Use when")
    print("                            you have >=50 minority samples and memory")
    print("                            allows storing the expanded training set.")
    print("  Approach B - Cost weight: No data augmentation; faster training.")
    print("                            Preferred for XGBoost/tree models and when")
    print("                            SMOTE over-fitting is a concern.")
    print("  Approach C - Pipeline   : CV-safe imblearn Pipeline (scaler+SMOTE+RF).")
    print("                            Use with cross_validate for unbiased eval.")

    # --- Approach C: imblearn Pipeline (scaler -> SMOTE -> classifier) ----------
    print("\n-- Approach C: imblearn Pipeline (CV-safe, no leakage) ---------------")
    smote_cv    = SMOTE(sampling_strategy=0.2, k_neighbors=k, random_state=RANDOM_STATE)
    rf_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  smote_cv),
        ("clf",    RandomForestClassifier(
            n_estimators = 100,
            class_weight = "balanced",
            n_jobs       = -1,
            random_state = RANDOM_STATE,
        )),
    ])
    print("[INFO] Pipeline: StandardScaler -> SMOTE(0.2) -> RandomForest(100 trees) [OK]")
    print("       Pass to cross_validate(rf_pipeline, X_train, y_train) for CV eval.")

    return (X_train_smote, y_train_smote, rf_balanced, xgb_balanced_params,
            scaler, X_train_scaled, X_test_scaled, rf_pipeline)


# =============================================================================
# SECTION 6 -- MODEL EVALUATION METRICS
# =============================================================================

def evaluate_models(rf_pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    5-fold stratified CV + holdout evaluation using the imblearn Pipeline.
    The Pipeline handles scaler + SMOTE internally so CV folds never leak.
    Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
    """

    print("\n" + "=" * 70)
    print("  SECTION 6 - MODEL EVALUATION METRICS")
    print("=" * 70)

    # --- 5-fold Stratified Cross-Validation -----------------------------------
    print("\n-- 5-fold Stratified Cross-Validation --------------------------------")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "precision" : "precision",
        "recall"    : "recall",
        "f1"        : "f1",
        "roc_auc"   : "roc_auc",
        "pr_auc"    : "average_precision",
    }

    print("[INFO] Running cross_validate — 5 folds x 5 metrics (may take ~2 min) ...")
    cv_results = cross_validate(
        rf_pipeline, X_train, y_train,
        cv                 = cv,
        scoring            = scoring,
        return_train_score = False,
        n_jobs             = -1,
        error_score        = "raise",
    )

    print("\n[INFO] CV results  (mean +/- std across 5 folds):")
    cv_labels = {
        "test_precision" : "Precision",
        "test_recall"    : "Recall",
        "test_f1"        : "F1-score",
        "test_roc_auc"   : "ROC-AUC",
        "test_pr_auc"    : "PR-AUC  (primary — imbalance)",
    }
    for key, label in cv_labels.items():
        scores = cv_results[key]
        print(f"  {label:<34} : {scores.mean():.4f}  +/- {scores.std():.4f}")

    # --- Holdout Test Set Evaluation ------------------------------------------
    print("\n-- Holdout Test Set Evaluation ---------------------------------------")
    print("[INFO] Fitting pipeline on full X_train ...")
    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_prob)
    pr_auc     = average_precision_score(y_test, y_prob)
    cm         = confusion_matrix(y_test, y_pred)

    print(f"\n[INFO] Holdout metrics:")
    print(f"  Precision                          : {precision:.4f}")
    print(f"  Recall                             : {recall:.4f}")
    print(f"  F1-score                           : {f1:.4f}")
    print(f"  ROC-AUC                            : {roc_auc:.4f}")
    print(f"  PR-AUC  (primary for imbalance)    : {pr_auc:.4f}")

    print(f"\n[INFO] Confusion Matrix:")
    print(f"  [[TN={cm[0,0]:>6,}  FP={cm[0,1]:>5,}]")
    print(f"   [FN={cm[1,0]:>6,}  TP={cm[1,1]:>5,}]]")

    print(f"\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legit", "Fraud"],
                                zero_division=0))

    # --- Evaluation plots: Confusion Matrix + PR Curve + ROC Curve -----------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), facecolor=CHARCOAL)
    fig.suptitle("Model Evaluation — Holdout Test Set", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)

    # Panel 1 — Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt=",d", cmap="RdYlGn",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
                ax=axes[0], linewidths=0.5, linecolor=CHARCOAL,
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Panel 2 — Precision-Recall curve
    prec_pts, rec_pts, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec_pts, prec_pts, color=ACCENT_FR, linewidth=2.2)
    axes[1].fill_between(rec_pts, prec_pts, alpha=0.2, color=ACCENT_FR)
    axes[1].axhline(float(y_test.mean()), color=ACCENT_LG, linestyle="--",
                    linewidth=1.4,
                    label=f"Baseline  (fraud rate {float(y_test.mean()):.4f})")
    axes[1].set_title(f"Precision-Recall Curve  (PR-AUC={pr_auc:.4f})",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)
    axes[1].grid(True)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.05)

    # Panel 3 — ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[2].plot(fpr, tpr, color="#7B68EE", linewidth=2.2,
                 label=f"RF  (AUC={roc_auc:.4f})")
    axes[2].fill_between(fpr, tpr, alpha=0.2, color="#7B68EE")
    axes[2].plot([0, 1], [0, 1], color=GRID_COLOR, linestyle="--", linewidth=1.2,
                 label="Random baseline")
    axes[2].set_title(f"ROC Curve  (AUC={roc_auc:.4f})",
                      fontsize=11, fontweight="bold")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(fontsize=9)
    axes[2].grid(True)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight",
                facecolor=CHARCOAL)
    print("\n[INFO] Evaluation plot saved -> model_evaluation.png")
    plt.close()

    return {
        "cv_results"       : cv_results,
        "precision"        : precision,
        "recall"           : recall,
        "f1"               : f1,
        "roc_auc"          : roc_auc,
        "pr_auc"           : pr_auc,
        "confusion_matrix" : cm,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    DATA_PATH = "creditcard.csv"

    # 1. Load & verify
    df = load_and_verify(DATA_PATH)

    # 2. EDA on the full clean dataframe (visualisation only — no stats computed)
    run_eda(df)

    # FIX 2: split FIRST on raw features, THEN engineer
    # 3. Stratified split on raw df (no engineered cols yet)
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Feature engineering — fit on train, transform test with train stats
    X_train, X_test, train_stats = engineer_features(X_train, X_test)

    # 5. Imbalance handling (SMOTE uses dynamic k; RF/XGB configs returned)
    (X_train_smote, y_train_smote, rf_balanced, xgb_params,
     scaler, X_train_scaled, X_test_scaled, rf_pipeline) = handle_imbalance(
        X_train, X_test, y_train
    )

    # 6. Evaluation metrics (5-fold CV + holdout — uses imblearn Pipeline)
    eval_metrics = evaluate_models(rf_pipeline, X_train, y_train, X_test, y_test)

    # Final artefact summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE - ARTEFACT SUMMARY")
    print("=" * 70)
    print(f"\n  X_train (eng. features) : {X_train.shape}")
    print(f"  X_test  (eng. features) : {X_test.shape}")
    print(f"  y_train                 : {y_train.shape}  | fraud={y_train.sum()}")
    print(f"  y_test                  : {y_test.shape}   | fraud={y_test.sum()}")
    print(f"  X_train_smote           : {X_train_smote.shape}")
    print(f"  y_train_smote           : {y_train_smote.shape} | fraud={int(y_train_smote.sum())}")
    print(f"\n  Saved plots:")
    print("    - eda_dashboard.png")
    print("    - engineered_features.png")
    print("    - smote_balance.png")
    print("    - model_evaluation.png")
    print(f"\n  Ready-to-train classifiers:")
    print("    - rf_balanced  (RandomForestClassifier, class_weight='balanced')")
    print("    - rf_pipeline  (ImbPipeline: scaler -> SMOTE(0.2) -> RF)")
    print("    - xgb_params   (XGBoost config dict, scale_pos_weight set)")
    print(f"\n  Evaluation metrics (holdout):")
    print(f"    - Precision : {eval_metrics['precision']:.4f}")
    print(f"    - Recall    : {eval_metrics['recall']:.4f}")
    print(f"    - F1-score  : {eval_metrics['f1']:.4f}")
    print(f"    - ROC-AUC   : {eval_metrics['roc_auc']:.4f}")
    print(f"    - PR-AUC    : {eval_metrics['pr_auc']:.4f}  (primary for imbalance)")
    print("\n[DONE] All steps completed successfully.\n")

    return {
        "X_train"         : X_train,
        "X_test"          : X_test,
        "y_train"         : y_train,
        "y_test"          : y_test,
        "train_stats"     : train_stats,
        "X_train_scaled"  : X_train_scaled,
        "X_test_scaled"   : X_test_scaled,
        "X_train_smote"   : X_train_smote,
        "y_train_smote"   : y_train_smote,
        "scaler"          : scaler,
        "rf_balanced"     : rf_balanced,
        "rf_pipeline"     : rf_pipeline,
        "xgb_params"      : xgb_params,
        "eval_metrics"    : eval_metrics,
    }


if __name__ == "__main__":
    artefacts = main()
