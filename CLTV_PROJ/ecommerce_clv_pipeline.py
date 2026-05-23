"""
E-Commerce CLV Data Pipeline
==============================
Senior Data Scientist Task — Customer Lifetime Value (CLV) Feature Engineering

Pipeline Steps:
  1. Load & inspect raw transactional data
  2. Remove bad data (negative quantities/prices, cancelled orders)
  3. Build RFM-style customer summary (Frequency, Recency, T, Monetary Value)
  4. Use lifetimes.utils.summary_data_from_transaction_data() to build
     the BG/NBD-ready summary table
  5. Quality checks — one row per customer, no missing/duplicate entries

Requirements:
  pip install pandas lifetimes

Dataset expected columns:
  InvoiceNo, StockCode, Description, Quantity,
  InvoiceDate, UnitPrice, CustomerID, Country
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
RAW_FILE = "data.csv"           # ← update path if needed
OUTPUT_FILE = "customer_summary.csv"

print("=" * 60)
print("STEP 1 — Loading raw data")
print("=" * 60)

df = pd.read_csv(RAW_FILE, encoding="latin-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"  Raw rows       : {len(df):,}")
print(f"  Raw columns    : {list(df.columns)}")
print(f"  Date range     : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"  Unique customers (raw, incl. NaN): {df['CustomerID'].nunique(dropna=False)}")

# ─────────────────────────────────────────────
# 2. REMOVE BAD DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Cleaning bad data")
print("=" * 60)

initial_rows = len(df)

# 2a. Drop rows with no CustomerID (can't attribute a customer)
df = df.dropna(subset=["CustomerID"])
print(f"  Dropped (no CustomerID)    : {initial_rows - len(df):,} rows")

# 2b. Remove cancelled orders (InvoiceNo starts with 'C')
mask_cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
n_cancelled = mask_cancelled.sum()
df = df[~mask_cancelled]
print(f"  Dropped (cancelled orders) : {n_cancelled:,} rows")

# 2c. Remove negative or zero quantities
mask_neg_qty = df["Quantity"] <= 0
n_neg_qty = mask_neg_qty.sum()
df = df[~mask_neg_qty]
print(f"  Dropped (Quantity ≤ 0)     : {n_neg_qty:,} rows")

# 2d. Remove negative or zero unit prices
mask_neg_price = df["UnitPrice"] <= 0
n_neg_price = mask_neg_price.sum()
df = df[~mask_neg_price]
print(f"  Dropped (UnitPrice ≤ 0)    : {n_neg_price:,} rows")

# 2e. Normalise CustomerID to integer
df["CustomerID"] = df["CustomerID"].astype(int)

print(f"\n  ✓ Clean rows remaining     : {len(df):,}")
print(f"  ✓ Unique customers (clean) : {df['CustomerID'].nunique():,}")

# ─────────────────────────────────────────────
# 3. MANUAL CUSTOMER SUMMARY  (for reference)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Manual RFM customer summary")
print("=" * 60)

# Add revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Define observation end date (last invoice in dataset)
observation_end = df["InvoiceDate"].max()

# Order-level aggregation (one row per invoice per customer)
orders = (
    df.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])
    ["Revenue"]
    .sum()
    .reset_index()
)

# Customer-level summary
manual_summary = (
    orders.groupby("CustomerID")
    .agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        frequency_raw=("InvoiceNo", "nunique"),   # total orders
        monetary_value=("Revenue", "mean"),        # avg order value
    )
    .reset_index()
)

# Recency  = days between first and last purchase
manual_summary["recency_days"] = (
    manual_summary["last_purchase"] - manual_summary["first_purchase"]
).dt.days

# T (age)  = days between first purchase and observation end
manual_summary["T_days"] = (
    observation_end - manual_summary["first_purchase"]
).dt.days

# frequency in lifetimes convention = number of REPEAT purchases
# (first purchase is not counted, so frequency = total_orders - 1)
manual_summary["frequency"] = manual_summary["frequency_raw"] - 1

manual_summary = manual_summary[[
    "CustomerID",
    "frequency",
    "recency_days",
    "T_days",
    "monetary_value",
    "first_purchase",
    "last_purchase",
]]

print(f"  Manual summary rows   : {len(manual_summary):,}")
print(f"  Sample (top 5):")
print(manual_summary.head().to_string(index=False))

# ─────────────────────────────────────────────
# 4. USE LIFETIMES LIBRARY (official BG/NBD format)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Building summary via lifetimes library")
print("=" * 60)

# summary_data_from_transaction_data expects:
#   transactions  : DataFrame with one row per transaction
#   customer_id_col, datetime_col, monetary_value_col
#   observation_period_end : end of observation window
#   freq : time unit for recency/T (default 'D' = days)

lifetimes_summary = summary_data_from_transaction_data(
    transactions=df,
    customer_id_col="CustomerID",
    datetime_col="InvoiceDate",
    monetary_value_col="Revenue",
    observation_period_end=observation_end,
    freq="D",        # recency and T expressed in days
)

# lifetimes returns: frequency, recency, T, monetary_value
# frequency = number of REPEAT transactions (total - 1)
# recency   = time between first and last transaction (days)
# T         = age of customer / time since first transaction (days)
# monetary_value = average transaction value (repeat transactions only)

print(f"  Lifetimes summary rows : {len(lifetimes_summary):,}")
print(f"  Columns                : {list(lifetimes_summary.columns)}")
print(f"\n  Sample (top 5):")
print(lifetimes_summary.head().to_string())
print(f"\n  Descriptive stats:")
print(lifetimes_summary.describe().round(2).to_string())

# ─────────────────────────────────────────────
# 5. QUALITY CHECKS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Quality checks")
print("=" * 60)

issues_found = 0

# 5a. Uniqueness — one row per customer
n_customers = len(lifetimes_summary)
n_unique_idx = lifetimes_summary.index.nunique()
if n_customers == n_unique_idx:
    print(f"  ✓ Uniqueness     : {n_customers:,} customers, all unique CustomerIDs")
else:
    print(f"  ✗ DUPLICATE CustomerIDs detected! rows={n_customers}, unique={n_unique_idx}")
    issues_found += 1

# 5b. Missing values
null_counts = lifetimes_summary.isnull().sum()
if null_counts.sum() == 0:
    print(f"  ✓ Missing values : none")
else:
    print(f"  ✗ Missing values found:\n{null_counts[null_counts > 0]}")
    issues_found += 1

# 5c. Non-negative constraints
neg_freq = (lifetimes_summary["frequency"] < 0).sum()
neg_rec  = (lifetimes_summary["recency"] < 0).sum()
neg_T    = (lifetimes_summary["T"] < 0).sum()
neg_mv   = (lifetimes_summary["monetary_value"] < 0).sum()

if neg_freq + neg_rec + neg_T + neg_mv == 0:
    print(f"  ✓ Non-negative   : frequency, recency, T, monetary_value all ≥ 0")
else:
    print(f"  ✗ Negative values: freq={neg_freq}, recency={neg_rec}, T={neg_T}, mv={neg_mv}")
    issues_found += 1

# 5d. Recency ≤ T (customer can't have repeat purchase before being acquired)
violated = (lifetimes_summary["recency"] > lifetimes_summary["T"]).sum()
if violated == 0:
    print(f"  ✓ Recency ≤ T    : all customers satisfy recency ≤ T")
else:
    print(f"  ✗ Recency > T violation in {violated:,} rows — investigate!")
    issues_found += 1

# 5e. Customers with zero repeat purchases (one-time buyers)
one_time_buyers = (lifetimes_summary["frequency"] == 0).sum()
pct = 100 * one_time_buyers / n_customers
print(f"  ℹ One-time buyers: {one_time_buyers:,} ({pct:.1f}%) — expected for e-commerce")

# 5f. Fix: fill any unexpected NaN monetary_value with 0
if lifetimes_summary["monetary_value"].isnull().any():
    lifetimes_summary["monetary_value"] = lifetimes_summary["monetary_value"].fillna(0)
    print("  ↳ Filled NaN monetary_value with 0")

if issues_found == 0:
    print(f"\n  ✅ All quality checks passed — dataset is clean and ready for BG/NBD modelling")
else:
    print(f"\n  ⚠️  {issues_found} issue(s) detected — review output above")

# ─────────────────────────────────────────────
# 6. SAVE OUTPUT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Saving final customer summary")
print("=" * 60)

lifetimes_summary.to_csv(OUTPUT_FILE, index=True)   # CustomerID is the index
print(f"  ✓ Saved '{OUTPUT_FILE}'  ({len(lifetimes_summary):,} customers × {lifetimes_summary.shape[1]} features)")
print(f"  Columns: {list(lifetimes_summary.columns)}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("""
Next Steps — BG/NBD + Gamma-Gamma CLV Modelling:

    from lifetimes import BetaGeoFitter, GammaGammaFitter

    # Fit BG/NBD model (purchase frequency)
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(df['frequency'], df['recency'], df['T'])

    # Fit Gamma-Gamma model (monetary value) — repeat buyers only
    repeat = df[df['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(repeat['frequency'], repeat['monetary_value'])

    # Predict 12-month CLV
    clv = ggf.customer_lifetime_value(
        bgf, df['frequency'], df['recency'], df['T'],
        df['monetary_value'], time=12, freq='D', discount_rate=0.01
    )
""")
