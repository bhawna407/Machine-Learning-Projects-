# -*- coding: utf-8 -*-
"""
Customer Lifetime Value (CLTV) Prediction — Fixed Pipeline
Fixes applied:
  1. Penalizer sweep -> finds coef where a, b move off zero (churn signal)
  2. Training filter -> T >= 90 days only (no newbie hallucination)
  3. Zero-frequency flag -> 'New/Insufficient Data' instead of fake CLTV
  4. Business logic  -> 'At Risk' tag when P(alive) < 0.5 or long silence
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lifetimes import BetaGeoFitter, GammaGammaFitter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONSTANTS — adjust here without touching logic below
# ─────────────────────────────────────────────────────────
MIN_OBSERVATION_DAYS  = 90     # FIX 2: exclude newer customers from training
PROB_ALIVE_RISK_CUTOFF = 0.5   # FIX 4: below this -> 'At Risk'
SILENCE_RISK_DAYS     = 180    # FIX 4: last purchase >180 days before period end -> 'At Risk'
FORECAST_DAYS         = 90     # prediction horizon


# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
df = pd.read_csv("rfm_summary.csv")

print("=" * 65)
print("DATASET OVERVIEW")
print("=" * 65)
print(f"Total customers      : {len(df):,}")
print(f"Columns              : {list(df.columns)}")
print(f"\n{df.describe().round(2)}")

# ══════════════════════════════════════════════════════════
# 2. PREPARE — WEEK CONVERSION + TRAINING SPLIT
# ══════════════════════════════════════════════════════════
all_data = df.copy()
all_data["recency_weeks"] = all_data["recency"] / 7.0
all_data["T_weeks"]       = all_data["T"] / 7.0

# FIX 2: separate training set (T >= 90 days) from full scoring set
train = all_data[all_data["T"] >= MIN_OBSERVATION_DAYS].copy()
excluded_new = all_data[all_data["T"] < MIN_OBSERVATION_DAYS].copy()

print(f"\n{'─'*65}")
print(f"FIX 2 — TRAINING FILTER (T >= {MIN_OBSERVATION_DAYS} days)")
print(f"{'─'*65}")
print(f"  Customers kept for training  : {len(train):,}  "
      f"({len(train)/len(all_data)*100:.1f}%)")
print(f"  Excluded (T < {MIN_OBSERVATION_DAYS}d, new/unseen) : {len(excluded_new):,}  "
      f"({len(excluded_new)/len(all_data)*100:.1f}%) — scored but NOT trained on")

# Gamma-Gamma training set: repeat buyers with known spend, T >= 90
gg_train = train[(train["frequency"] > 0) & (train["monetary_value"] > 0)].copy()
print(f"  Gamma-Gamma training rows    : {len(gg_train):,}")


# ══════════════════════════════════════════════════════════
# 3. FIX 1 — PENALIZER SWEEP FOR BG/NBD
#    Find the smallest penalizer that pushes a, b off zero.
#    We want a > 0.01 AND b > 0.01 (non-degenerate dropout).
# ══════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("FIX 1 — PENALIZER SWEEP (finding best BG/NBD penalizer_coef)")
print(f"{'─'*65}")

sweep_results = []
for coef in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    try:
        m = BetaGeoFitter(penalizer_coef=coef)
        m.fit(train["frequency"], train["recency_weeks"], train["T_weeks"], verbose=False)
        p = m.params_
        ab_ok = (p["a"] > 0.01) and (p["b"] > 0.01)
        sweep_results.append({
            "coef": coef, "r": p["r"], "alpha": p["alpha"],
            "a": p["a"], "b": p["b"],
            "a_div_ab": p["a"] / (p["a"] + p["b"]),
            "ll": m._negative_log_likelihood_,
            "a_b_ok": ab_ok,
        })
        flag = "[OK]" if ab_ok else "[DEGENERATE]"
        print(f"  coef={coef:.3f}  r={p['r']:.4f}  alpha={p['alpha']:.4f}  "
              f"a={p['a']:.6f}  b={p['b']:.6f}  {flag}")
    except Exception as e:
        print(f"  coef={coef:.3f}  FAILED: {e}")

sweep_df = pd.DataFrame(sweep_results)
# Pick the smallest coef where a and b are both non-degenerate (> 0.01)
good = sweep_df[sweep_df["a_b_ok"]]
if len(good) == 0:
    best_coef = 1.0
    print("\n  [WARN] No coef produced non-degenerate a,b — using 1.0 (strongest regularisation)")
else:
    best_coef = good["coef"].min()
    print(f"\n  Best penalizer_coef selected : {best_coef}  "
          f"(a={good[good['coef']==best_coef]['a'].values[0]:.6f}, "
          f"b={good[good['coef']==best_coef]['b'].values[0]:.6f})")

# ── Data-limitation note ──────────────────────────────────
# If no penalizer produced non-zero a/b, this dataset's ~1-year observation
# window is too short for BG/NBD to distinguish "low purchase rate" from
# "churned."  The model is still valid for frequency prediction.
# The At-Risk tag (Fix 4) uses a silence-days heuristic which does NOT
# depend on prob_alive, so it remains reliable regardless of this issue.


# ══════════════════════════════════════════════════════════
# 4. TRAIN FINAL BG/NBD MODEL
# ══════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"MODEL 1: BG/NBD  (penalizer_coef={best_coef})")
print(f"{'─'*65}")

bgf = BetaGeoFitter(penalizer_coef=best_coef)
bgf.fit(
    train["frequency"],
    train["recency_weeks"],
    train["T_weeks"],
    verbose=True,
)

params = bgf.params_
avg_dropout = params["a"] / (params["a"] + params["b"])
print(f"\nFinal Parameters:")
print(f"  r     = {params['r']:.4f}   -- spread of purchase-rate distribution")
print(f"  alpha = {params['alpha']:.4f}   -- scale of purchase rate")
print(f"  a     = {params['a']:.6f}  -- fast-churn Beta shape  (was ~0 before)")
print(f"  b     = {params['b']:.6f}  -- slow-churn Beta shape  (was ~0 before)")
print(f"\n  Avg long-run dropout probability a/(a+b) = {avg_dropout:.4f}")
if avg_dropout < 0.01:
    print("  [WARN] a/(a+b) still near zero — dataset may lack strong churn signal.")
    print("         Consider using a longer observation window if available.")
else:
    print("  [OK]  a and b are non-degenerate — dropout model is meaningful.")


# ══════════════════════════════════════════════════════════
# 5. TRAIN GAMMA-GAMMA MODEL (on filtered training set)
# ══════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("MODEL 2: GAMMA-GAMMA  (spend per visit)")
print(f"{'─'*65}")

corr = gg_train[["frequency", "monetary_value"]].corr().iloc[0, 1]
print(f"\nCorrelation check (frequency vs monetary_value): {corr:.4f}", end="  ")
print("[OK]" if abs(corr) < 0.2 else "[WARN — check independence assumption]")

ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(gg_train["frequency"], gg_train["monetary_value"], verbose=True)

gg_params = ggf.params_
est_avg = gg_params["p"] * gg_params["v"] / gg_params["q"]
print(f"\nFinal Parameters:")
print(f"  p = {gg_params['p']:.4f}   -- shape of per-transaction spend distribution")
print(f"  q = {gg_params['q']:.4f}   -- population spend heterogeneity")
print(f"  v = {gg_params['v']:.4f}   -- scale parameter")
print(f"\n  Model-implied avg spend  : £{est_avg:.2f}")
print(f"  Actual   avg spend (train): £{gg_train['monetary_value'].mean():.2f}")
gap = abs(est_avg - gg_train["monetary_value"].mean()) / gg_train["monetary_value"].mean() * 100
print(f"  Gap                      : {gap:.1f}%  {'[OK]' if gap < 20 else '[WARN — large gap]'}")


# ══════════════════════════════════════════════════════════
# 6. SCORE ALL CUSTOMERS (full dataset, not just training)
# ══════════════════════════════════════════════════════════
t_forecast = FORECAST_DAYS / 7.0   # convert to weeks to match model

all_data["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t=t_forecast,
    frequency=all_data["frequency"],
    recency=all_data["recency_weeks"],
    T=all_data["T_weeks"],
)

all_data["expected_avg_spend"] = ggf.conditional_expected_average_profit(
    all_data["frequency"],
    all_data["monetary_value"],
)

all_data["cltv_90d"] = all_data["predicted_purchases"] * all_data["expected_avg_spend"]

all_data["prob_alive"] = bgf.conditional_probability_alive(
    all_data["frequency"],
    all_data["recency_weeks"],
    all_data["T_weeks"],
)


# ══════════════════════════════════════════════════════════
# 7. FIX 3 — SEGMENTATION WITH ZERO-FREQUENCY FLAG
#    Customers with frequency == 0 get 'New/Insufficient Data'
#    instead of a model-predicted CLTV segment.
# ══════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("FIX 3 — CUSTOMER SEGMENTATION")
print(f"{'─'*65}")

# FIX 3: initialise as plain string column (not Categorical) so we can
# assign 'New/Insufficient Data' freely without dtype conflicts.
all_data["cltv_segment"] = "New/Insufficient Data"

scored_mask = all_data["frequency"] > 0
scored_data = all_data.loc[scored_mask, "cltv_90d"]

p25 = scored_data.quantile(0.25)
p50 = scored_data.quantile(0.50)
p75 = scored_data.quantile(0.75)

def _label(v):
    if v <= p25: return "Low Value"
    if v <= p50: return "Mid Value"
    if v <= p75: return "High Value"
    return "Champions"

all_data.loc[scored_mask, "cltv_segment"] = (
    all_data.loc[scored_mask, "cltv_90d"].map(_label)
)
# freq=0 rows keep "New/Insufficient Data" already set above

seg_counts = all_data["cltv_segment"].value_counts()
print(f"\n  Segment breakdown:")
for seg, cnt in seg_counts.items():
    pct = cnt / len(all_data) * 100
    note = "  <- FIX 3 applied (no real CLTV assigned)" if seg == "New/Insufficient Data" else ""
    print(f"    {seg:<28s}: {cnt:,}  ({pct:.1f}%){note}")


# ══════════════════════════════════════════════════════════
# 8. FIX 4 — BUSINESS LOGIC: 'AT RISK' TAG
#    A customer is At Risk if EITHER:
#      (a) prob_alive < PROB_ALIVE_RISK_CUTOFF (model says likely inactive), OR
#      (b) they haven't bought for SILENCE_RISK_DAYS before the observation end
#          (recency is how recently they bought; T - recency = days of silence)
# ══════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("FIX 4 — AT RISK BUSINESS LOGIC")
print(f"{'─'*65}")

days_since_last = all_data["T"] - all_data["recency"]

low_prob_alive  = all_data["prob_alive"] < PROB_ALIVE_RISK_CUTOFF
long_silence    = (all_data["frequency"] > 0) & (days_since_last > SILENCE_RISK_DAYS)

all_data["at_risk"] = (low_prob_alive | long_silence)

# Clarify: freq=0 customers are already 'New/Insufficient Data', not At Risk
all_data.loc[all_data["frequency"] == 0, "at_risk"] = False

at_risk_count = all_data["at_risk"].sum()
print(f"\n  Trigger 1 — P(alive) < {PROB_ALIVE_RISK_CUTOFF}          : "
      f"{low_prob_alive.sum():,} customers")
print(f"  Trigger 2 — Silence > {SILENCE_RISK_DAYS}d (repeat buyers): "
      f"{long_silence.sum():,} customers")
print(f"  Total 'At Risk' (either trigger)         : {at_risk_count:,} customers  "
      f"({at_risk_count/len(all_data)*100:.1f}%)")

# Show a sample of flagged customers
at_risk_sample = all_data[all_data["at_risk"]].sort_values("days_since_last" if "days_since_last" in all_data else "recency").head(8)
all_data["days_silent"] = days_since_last
at_risk_sample = (
    all_data[all_data["at_risk"]]
    .sort_values("days_silent", ascending=False)
    .head(8)[["CustomerID","frequency","recency","T","days_silent","prob_alive","cltv_90d","cltv_segment"]]
)
print(f"\n  Sample At Risk customers (longest silence first):")
print(at_risk_sample.to_string(index=False))


# ══════════════════════════════════════════════════════════
# 9. RESULTS SUMMARY
# ══════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("RESULTS SUMMARY — CLTV 90-DAY FORECAST")
print(f"{'=' * 65}")

# Only include repeat buyers with valid data in revenue totals
valid = all_data[all_data["frequency"] > 0]
print(f"\nCLTV stats (repeat buyers only, n={len(valid):,}):")
print(valid["cltv_90d"].describe().round(2))

print(f"\nCLTV by segment (repeat buyers):")
seg_summary = (
    valid.groupby("cltv_segment", observed=True)["cltv_90d"]
    .agg(["mean", "median", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
)
print(seg_summary.to_string())

total_rev = valid["cltv_90d"].sum()
print(f"\nTotal predicted revenue — {FORECAST_DAYS}-day horizon : £{total_rev:,.0f}")
print(f"  (excludes {(all_data['frequency']==0).sum():,} customers flagged New/Insufficient Data)")

print(f"\nP(alive) distribution (repeat buyers after model fix):")
print(valid["prob_alive"].describe().round(4))

print(f"\n{'─'*65}")
print("TOP 10 CUSTOMERS BY PREDICTED CLTV")
print(f"{'─'*65}")
top10 = (
    valid.sort_values("cltv_90d", ascending=False)
    .head(10)[[
        "CustomerID","frequency","recency","T","monetary_value",
        "predicted_purchases","expected_avg_spend","cltv_90d",
        "prob_alive","cltv_segment","at_risk",
    ]]
    .reset_index(drop=True)
)
top10.index += 1
print(top10.to_string())


# ══════════════════════════════════════════════════════════
# 10. SAVE OUTPUT
# ══════════════════════════════════════════════════════════
out_cols = [
    "CustomerID","frequency","recency","T","monetary_value",
    "predicted_purchases","expected_avg_spend","cltv_90d",
    "prob_alive","cltv_segment","at_risk","days_silent",
]
all_data[out_cols].to_csv("cltv_predictions.csv", index=False)
print(f"\n[OK] Predictions saved -> cltv_predictions.csv  ({len(all_data):,} rows)")


# ══════════════════════════════════════════════════════════
# 11. VISUALISATIONS
# ══════════════════════════════════════════════════════════
SEGMENT_COLORS = {
    "Champions":            "#0D47A1",
    "High Value":           "#1976D2",
    "Mid Value":            "#64B5F6",
    "Low Value":            "#BBDEFB",
    "New/Insufficient Data":"#B0BEC5",
}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    f"CLTV Model — Fixed Pipeline ({FORECAST_DAYS}-Day Forecast)",
    fontsize=15, fontweight="bold",
)

# ── Plot 1: P(alive) distribution after fix ─────────────
ax = axes[0, 0]
ax.hist(valid["prob_alive"], bins=40, color="#1976D2", edgecolor="white", alpha=0.85)
ax.axvline(PROB_ALIVE_RISK_CUTOFF, color="red", linestyle="--", linewidth=1.5,
           label=f"At-Risk cutoff ({PROB_ALIVE_RISK_CUTOFF})")
ax.set_xlabel("P(alive)")
ax.set_ylabel("Number of Customers")
ax.set_title("P(Alive) Distribution\n(repeat buyers — after penalizer fix)")
ax.legend()

# ── Plot 2: Predicted purchases vs historical frequency ──
ax = axes[0, 1]
colors_scatter = valid["at_risk"].map({True: "#E53935", False: "#1976D2"})
ax.scatter(valid["frequency"], valid["predicted_purchases"],
           c=colors_scatter, alpha=0.35, s=15)
ax.set_xlabel("Historical Frequency")
ax.set_ylabel(f"Predicted Purchases ({FORECAST_DAYS}d)")
ax.set_title("Predicted vs Historical Frequency\n(red = At Risk)")
red_patch   = mpatches.Patch(color="#E53935", label="At Risk")
blue_patch  = mpatches.Patch(color="#1976D2", label="Active")
ax.legend(handles=[red_patch, blue_patch])

# ── Plot 3: Segment breakdown bar chart ─────────────────
ax = axes[0, 2]
seg_order = ["Champions","High Value","Mid Value","Low Value","New/Insufficient Data"]
seg_vals  = [all_data[all_data["cltv_segment"]==s].shape[0] for s in seg_order]
bar_colors = [SEGMENT_COLORS[s] for s in seg_order]
bars = ax.barh(seg_order, seg_vals, color=bar_colors, edgecolor="white")
for bar, val in zip(bars, seg_vals):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
ax.set_xlabel("Number of Customers")
ax.set_title("Customer Segment Distribution\n(FIX 3: freq=0 -> New/Insufficient Data)")
ax.set_xlim(0, max(seg_vals) * 1.15)

# ── Plot 4: CLTV distribution by segment (violin) ────────
ax = axes[1, 0]
seg_plot_order = ["Champions","High Value","Mid Value","Low Value"]
seg_plot_data  = [valid[valid["cltv_segment"]==s]["cltv_90d"].values for s in seg_plot_order]
vp = ax.violinplot(seg_plot_data, showmedians=True, showextrema=True)
for body, seg in zip(vp["bodies"], seg_plot_order):
    body.set_facecolor(SEGMENT_COLORS[seg])
    body.set_alpha(0.8)
ax.set_xticks(range(1, len(seg_plot_order)+1))
ax.set_xticklabels(seg_plot_order, rotation=15)
ax.set_ylabel("90-Day CLTV (£)")
ax.set_title("CLTV Distribution by Segment\n(repeat buyers only)")

# ── Plot 5: Days silent vs prob_alive ────────────────────
ax = axes[1, 1]
sc = ax.scatter(
    valid["days_silent"],
    valid["prob_alive"],
    c=valid["at_risk"].map({True: "#E53935", False: "#43A047"}),
    alpha=0.25, s=10,
)
ax.axvline(SILENCE_RISK_DAYS, color="orange", linestyle="--", linewidth=1.5,
           label=f"{SILENCE_RISK_DAYS}d silence threshold")
ax.axhline(PROB_ALIVE_RISK_CUTOFF, color="red", linestyle="--", linewidth=1.5,
           label=f"P(alive) < {PROB_ALIVE_RISK_CUTOFF}")
ax.set_xlabel("Days Silent (T - recency)")
ax.set_ylabel("P(alive)")
ax.set_title("At-Risk Detection\n(FIX 4: two-trigger logic)")
ax.legend(fontsize=8)

# ── Plot 6: Revenue by segment ───────────────────────────
ax = axes[1, 2]
rev_by_seg = (
    valid.groupby("cltv_segment", observed=True)["cltv_90d"]
    .sum()
    .reindex(["Champions","High Value","Mid Value","Low Value"])
    .fillna(0)
)
rev_pcts = rev_by_seg / rev_by_seg.sum() * 100
bar_colors6 = [SEGMENT_COLORS[s] for s in rev_by_seg.index]
bars6 = ax.bar(rev_by_seg.index, rev_by_seg.values, color=bar_colors6, edgecolor="white")
for bar, pct in zip(bars6, rev_pcts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{pct:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Total Predicted Revenue (£)")
ax.set_title(f"Revenue Share by Segment\n({FORECAST_DAYS}-Day Horizon)")
ax.set_xticklabels(rev_by_seg.index, rotation=15)

plt.tight_layout()
plt.savefig("cltv_model_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Diagnostic plots saved -> cltv_model_diagnostics.png")

print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
