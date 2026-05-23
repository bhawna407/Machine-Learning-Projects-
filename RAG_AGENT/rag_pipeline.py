"""
RAG Agent - Day 1 Pipeline  (corrected)
Olist E-Commerce Dataset

Bugs fixed vs v1:
  - Chunking now uses actual BPE tokenizer (not 4-chars/token heuristic)
    -> chunks target 150-200 BPE tokens, safely within all-MiniLM-L6-v2 max of 256
  - Seller review scores: deduplicated per order before averaging (was inflated
    because one review was counted N times for N items in the order)
  - avg_order_value: now computed as total order revenue / unique orders (was
    averaging per-item revenue which understated multi-item orders)
  - Monthly totals: now computed from original delivered data (not sum of
    category sub-groups which double-counts cross-category orders)
  - Delivery report: renamed "Avg Delay vs ETA" to "Avg Days vs ETA
    (neg=early)" and added explicit "most-late% region" stat
  - Complaints filter: chained with & instead of fragile double-bracket indexing
"""

import json
import re
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\PC\Downloads\CLAUDE CODE\P03_RAG_AGENT\DAY_1")
DATA = BASE / "RAG AGENT DATA"
OUT  = BASE / "output"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BPE_MAX    = 256   # hard limit of all-MiniLM-L6-v2
CHUNK_MIN  = 75    # BPE tokens — small paragraphs under this get merged up
CHUNK_MAX  = 200   # BPE tokens — keep well under model's 256-token hard limit

print("=" * 70)
print("RAG AGENT - DAY 1 PIPELINE  (v2 corrected)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading datasets...")

orders    = pd.read_csv(DATA / "olist_orders_dataset.csv", parse_dates=[
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
])
items     = pd.read_csv(DATA / "olist_order_items_dataset.csv")
reviews   = pd.read_csv(DATA / "olist_order_reviews_dataset.csv")
sellers   = pd.read_csv(DATA / "olist_sellers_dataset.csv")
customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
products  = pd.read_csv(DATA / "olist_products_dataset.csv")
cat_trans = pd.read_csv(DATA / "product_category_name_translation.csv")

products = products.merge(cat_trans, on="product_category_name", how="left")
products["category"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
    .fillna("unknown")
)

master = (
    orders
    .merge(items,                              on="order_id",   how="left")
    .merge(products[["product_id", "category"]], on="product_id", how="left")
    .merge(sellers,                            on="seller_id",  how="left")
    .merge(customers,                          on="customer_id", how="left")
)
master["revenue"]             = master["price"] + master["freight_value"]
master["purchase_month"]      = master["order_purchase_timestamp"].dt.to_period("M")
master["days_to_deliver"]     = (
    master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
).dt.days
master["delivery_delay_days"] = (
    master["order_delivered_customer_date"] - master["order_estimated_delivery_date"]
).dt.days
print(f"   Master table: {len(master):,} rows")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — GENERATE STRUCTURED TEXT REPORTS
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Generating structured text reports...")

reports: dict[str, str] = {}


# ── Report 1: Monthly Sales Summary by Category ──────────────────────────
def report_monthly_sales() -> str:
    delivered = master[master["order_status"] == "delivered"].dropna(
        subset=["purchase_month", "category"]
    )

    # Per-category per-month metrics
    # avg_order_value: compute total revenue per order first, then average
    order_revenue = (
        delivered
        .groupby(["order_id", "purchase_month", "category"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "order_total"})
    )
    cat_df = (
        order_revenue
        .groupby(["purchase_month", "category"])
        .agg(
            total_revenue=("order_total", "sum"),
            num_orders=("order_id", "nunique"),
            avg_order_value=("order_total", "mean"),   # FIX: mean of per-order totals
        )
        .reset_index()
        .sort_values(["purchase_month", "total_revenue"], ascending=[True, False])
    )

    # Overall monthly totals — computed from raw delivered data, not summed from categories
    # FIX: avoids double-counting orders that contain items from multiple categories
    monthly_totals = (
        delivered
        .groupby("purchase_month")
        .agg(
            total_rev=("revenue", "sum"),
            total_orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    lines = [
        "MONTHLY SALES SUMMARY BY CATEGORY",
        "=" * 50,
        "This report covers delivered orders from the Olist e-commerce platform.",
        "Metrics: total revenue (BRL), number of unique orders, avg total order value.",
        "",
        "OVERALL MONTHLY TOTALS",
        "-" * 40,
    ]
    for _, row in monthly_totals.iterrows():
        lines.append(
            f"Month {row['purchase_month']}: Revenue BRL {row['total_rev']:,.2f}, "
            f"Orders {int(row['total_orders']):,}"
        )
    lines.append("")

    lines.append("TOP 3 CATEGORIES PER MONTH")
    lines.append("-" * 40)
    for month in cat_df["purchase_month"].unique():
        sub = cat_df[cat_df["purchase_month"] == month].head(3)
        lines.append(f"\nMonth {month}:")
        for _, row in sub.iterrows():
            lines.append(
                f"  Category '{row['category']}': Revenue BRL {row['total_revenue']:,.2f}, "
                f"Orders {int(row['num_orders'])}, "
                f"Avg Order Value BRL {row['avg_order_value']:,.2f}"
            )
    lines.append("")

    # All-time category ranking
    cat_summary = (
        delivered
        .groupby("category")
        .agg(
            total_revenue=("revenue", "sum"),
            num_orders=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    lines.append("ALL-TIME CATEGORY REVENUE RANKING")
    lines.append("-" * 40)
    for rank, (_, row) in enumerate(cat_summary.iterrows(), 1):
        lines.append(
            f"Rank {rank}: '{row['category']}' -- Revenue BRL {row['total_revenue']:,.2f}, "
            f"Orders {int(row['num_orders'])}"
        )

    return "\n".join(lines)


# ── Report 2: Seller Performance ─────────────────────────────────────────
def report_seller_performance() -> str:
    delivered = master[master["order_status"] == "delivered"].copy()

    seller_stats = (
        delivered
        .dropna(subset=["seller_id"])
        .groupby(["seller_id", "seller_state"])
        .agg(
            total_revenue=("revenue", "sum"),
            total_orders=("order_id", "nunique"),
            avg_delivery_days=("days_to_deliver", "mean"),
            avg_delay_days=("delivery_delay_days", "mean"),
        )
        .reset_index()
    )

    # FIX: dedup reviews per order before attributing to seller
    # Each review belongs to one order; one order may have items from multiple sellers.
    # Attribution: assign each order's review score to every seller in that order
    # (Olist limitation — reviews are order-level, not seller-level).
    # But we must NOT count the same order's review multiple times for the same seller.
    order_seller = (
        items[["order_id", "seller_id"]]
        .drop_duplicates()   # one row per (order, seller) pair
    )
    rev_scores = (
        reviews[["order_id", "review_score"]]
        .merge(order_seller, on="order_id")
        .groupby("seller_id")["review_score"]
        .mean()
        .reset_index()
        .rename(columns={"review_score": "avg_review_score"})
    )
    seller_stats = seller_stats.merge(rev_scores, on="seller_id", how="left")
    seller_stats = seller_stats.sort_values("total_revenue", ascending=False)

    top10    = seller_stats.head(10)
    bottom10 = seller_stats.tail(10)

    lines = [
        "SELLER PERFORMANCE REPORT",
        "=" * 50,
        "Analysis of seller performance: revenue, delivery speed, delays, customer ratings.",
        "Note: review scores are attributed at order level (Olist dataset limitation).",
        "",
        f"Total active sellers: {len(seller_stats):,}",
        f"Overall avg delivery days: {seller_stats['avg_delivery_days'].mean():.1f}",
        f"Overall avg review score: {seller_stats['avg_review_score'].mean():.2f} / 5.0",
        "",
        "TOP 10 SELLERS BY REVENUE",
        "-" * 40,
    ]
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        score_str = f"{row['avg_review_score']:.2f}" if pd.notna(row["avg_review_score"]) else "N/A"
        delay_str = f"{row['avg_delay_days']:.1f}" if pd.notna(row["avg_delay_days"]) else "N/A"
        lines.append(
            f"Rank {rank} | Seller {row['seller_id'][:8]}... | State: {row['seller_state']} | "
            f"Revenue: BRL {row['total_revenue']:,.2f} | Orders: {int(row['total_orders'])} | "
            f"Avg Delivery: {row['avg_delivery_days']:.1f} days | "
            f"Avg Delay vs ETA: {delay_str} days | Rating: {score_str}/5"
        )
    lines.append("")

    lines.append("BOTTOM 10 SELLERS BY REVENUE")
    lines.append("-" * 40)
    for rank, (_, row) in enumerate(bottom10.iterrows(), 1):
        score_str = f"{row['avg_review_score']:.2f}" if pd.notna(row["avg_review_score"]) else "N/A"
        lines.append(
            f"Bottom-{rank} | Seller {row['seller_id'][:8]}... | State: {row['seller_state']} | "
            f"Revenue: BRL {row['total_revenue']:,.2f} | Orders: {int(row['total_orders'])} | "
            f"Rating: {score_str}/5"
        )
    lines.append("")

    state_summary = (
        seller_stats
        .groupby("seller_state")
        .agg(
            num_sellers=("seller_id", "count"),
            total_revenue=("total_revenue", "sum"),
            avg_review=("avg_review_score", "mean"),
            avg_delivery=("avg_delivery_days", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    lines.append("SELLER PERFORMANCE BY STATE")
    lines.append("-" * 40)
    for _, row in state_summary.iterrows():
        lines.append(
            f"State {row['seller_state']}: {int(row['num_sellers'])} sellers, "
            f"Revenue BRL {row['total_revenue']:,.2f}, "
            f"Avg Rating {row['avg_review']:.2f}, "
            f"Avg Delivery {row['avg_delivery']:.1f} days"
        )

    return "\n".join(lines)


# ── Report 3: Order Delivery Performance by Region ────────────────────────
def report_delivery_performance() -> str:
    delivered = master[
        (master["order_status"] == "delivered") &
        master["days_to_deliver"].notna()
    ].copy()

    region_map = {
        "SP": "Southeast", "RJ": "Southeast", "MG": "Southeast", "ES": "Southeast",
        "PR": "South",     "SC": "South",     "RS": "South",
        "BA": "Northeast", "CE": "Northeast", "PE": "Northeast", "MA": "Northeast",
        "PB": "Northeast", "RN": "Northeast", "AL": "Northeast", "SE": "Northeast",
        "PI": "Northeast",
        "AM": "North",     "PA": "North",     "RO": "North",     "AC": "North",
        "RR": "North",     "AP": "North",     "TO": "North",
        "MT": "Midwest",   "GO": "Midwest",   "MS": "Midwest",   "DF": "Midwest",
    }
    delivered["region"] = delivered["customer_state"].map(region_map).fillna("Other")

    region_stats = (
        delivered
        .groupby(["region", "customer_state"])
        .agg(
            num_orders=("order_id", "nunique"),
            avg_delivery_days=("days_to_deliver", "mean"),
            median_delivery_days=("days_to_deliver", "median"),
            max_delivery_days=("days_to_deliver", "max"),
            avg_days_vs_eta=("delivery_delay_days", "mean"),
            pct_late=("delivery_delay_days", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
        .sort_values("avg_delivery_days", ascending=False)
    )

    region_summary = (
        delivered
        .groupby("region")
        .agg(
            num_orders=("order_id", "nunique"),
            avg_delivery_days=("days_to_deliver", "mean"),
            median_delivery_days=("days_to_deliver", "median"),
            avg_days_vs_eta=("delivery_delay_days", "mean"),     # FIX: renamed from "delay"
            pct_late=("delivery_delay_days", lambda x: (x > 0).mean() * 100),
        )
        .reset_index()
        .sort_values("avg_delivery_days", ascending=False)
    )

    worst_abs  = region_summary.iloc[0]
    best_abs   = region_summary.iloc[-1]
    # FIX: separately identify the region with the highest % of late orders
    worst_late = region_summary.sort_values("pct_late", ascending=False).iloc[0]

    lines = [
        "ORDER DELIVERY PERFORMANCE BY REGION",
        "=" * 50,
        "Analysis of delivery speed and on-time performance across Brazilian regions.",
        "Avg Days vs ETA: negative = package arrived BEFORE estimated date (ETA is conservative).",
        f"Total delivered orders analysed: {len(delivered):,}",
        "",
        "REGIONAL SUMMARY (sorted by avg delivery days, slowest first)",
        "-" * 60,
    ]
    for _, row in region_summary.iterrows():
        lines.append(
            f"Region: {row['region']} | Orders: {int(row['num_orders']):,} | "
            f"Avg Delivery: {row['avg_delivery_days']:.1f} days | "
            f"Median: {row['median_delivery_days']:.1f} days | "
            f"Avg Days vs ETA: {row['avg_days_vs_eta']:.1f} (neg=early) | "
            f"Late Orders: {row['pct_late']:.1f}%"
        )
    lines.append("")

    lines.append("STATE-LEVEL DELIVERY PERFORMANCE (slowest absolute time first)")
    lines.append("-" * 60)
    for _, row in region_stats.iterrows():
        lines.append(
            f"State {row['customer_state']} ({row['region']}) | "
            f"Orders: {int(row['num_orders']):,} | "
            f"Avg: {row['avg_delivery_days']:.1f} days | "
            f"Max: {int(row['max_delivery_days'])} days | "
            f"Avg Days vs ETA: {row['avg_days_vs_eta']:.1f} (neg=early) | "
            f"Late%: {row['pct_late']:.1f}%"
        )
    lines.append("")

    lines.append("DELIVERY PERFORMANCE EXTREMES")
    lines.append("-" * 40)
    lines.append(
        f"SLOWEST absolute delivery: {worst_abs['region']} region -- "
        f"avg {worst_abs['avg_delivery_days']:.1f} days."
    )
    lines.append(
        f"FASTEST absolute delivery: {best_abs['region']} region -- "
        f"avg {best_abs['avg_delivery_days']:.1f} days."
    )
    lines.append(
        f"MOST LATE ORDERS (% basis): {worst_late['region']} region -- "
        f"{worst_late['pct_late']:.1f}% of orders arrived after estimated date."
    )

    return "\n".join(lines)


# ── Report 4: Customer Complaint Themes ──────────────────────────────────
def report_customer_complaints() -> str:
    rev = reviews.merge(
        orders[["order_id", "order_status"]], on="order_id", how="left"
    )
    rev["review_comment_message"] = rev["review_comment_message"].fillna("").astype(str)
    rev["review_comment_title"]   = rev["review_comment_title"].fillna("").astype(str)

    score_dist = rev["review_score"].value_counts().sort_index()
    total = len(rev)

    lines = [
        "CUSTOMER COMPLAINT THEMES FROM REVIEWS",
        "=" * 50,
        "Analysis of customer satisfaction and complaint patterns from Olist order reviews.",
        "",
        f"Total reviews: {total:,}",
        "Review score distribution:",
    ]
    for score, count in score_dist.items():
        lines.append(f"  Score {score}/5: {count:,} reviews ({count/total*100:.1f}%)")
    lines.append("")

    neg_reviews = rev[rev["review_score"] <= 2].copy()
    pos_reviews = rev[rev["review_score"] >= 4].copy()
    lines.append(f"Negative reviews (score 1-2): {len(neg_reviews):,} ({len(neg_reviews)/total*100:.1f}%)")
    lines.append(f"Positive reviews (score 4-5): {len(pos_reviews):,} ({len(pos_reviews)/total*100:.1f}%)")
    lines.append(f"Neutral reviews  (score 3):   {len(rev[rev['review_score']==3]):,}")
    lines.append("")

    complaint_keywords = {
        "Late Delivery / Not Received": [
            "not received", "nao recebi", "ainda nao chegou", "nunca chegou",
            "prazo", "atrasado", "atraso", "demora", "demorou", "entrega atrasada",
            "not delivered", "late delivery", "delayed",
        ],
        "Wrong / Damaged Product": [
            "produto errado", "wrong product", "diferente", "danificado", "quebrado",
            "damaged", "broken", "different from", "veio diferente", "produto com defeito",
            "defeito",
        ],
        "Poor Product Quality": [
            "qualidade", "ruim", "pessima", "horrivel", "muito fraco",
            "poor quality", "bad quality", "cheap", "nao funciona", "nao funcionou",
            "nao presta", "lixo", "fake", "falsificado",
        ],
        "Seller Communication Issues": [
            "sem resposta", "nao respondeu", "ignorou", "no response",
            "atendimento", "customer service", "support", "vendedor", "seller",
        ],
        "Packaging Issues": [
            "embalagem", "packaging", "caixa amassada", "caixa aberta",
            "mal embalado", "badly packed", "box damaged",
        ],
        "Missing Items": [
            "faltou", "faltando", "missing", "incompleto", "nao vieram",
            "vieram so", "so vieram", "missing items",
        ],
        "Refund / Return Problems": [
            "reembolso", "refund", "devolucao", "return", "cancelamento",
            "cancelei", "nao recebi reembolso", "estorno",
        ],
    }

    lines.append("COMPLAINT THEME ANALYSIS (from negative reviews, score 1-2)")
    lines.append("-" * 60)
    all_neg_text = (
        neg_reviews["review_comment_message"] + " " +
        neg_reviews["review_comment_title"]
    ).str.lower()

    theme_counts = {}
    for theme, keywords in complaint_keywords.items():
        pattern = "|".join(re.escape(kw) for kw in keywords)
        theme_counts[theme] = all_neg_text.str.contains(pattern, na=False).sum()

    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(neg_reviews) * 100 if len(neg_reviews) else 0
        lines.append(f"  Theme '{theme}': {count:,} mentions ({pct:.1f}% of negative reviews)")
    lines.append("")

    lines.append("AVERAGE REVIEW SCORE BY ORDER STATUS")
    lines.append("-" * 40)
    status_scores = (
        rev.groupby("order_status")["review_score"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean")
    )
    for _, row in status_scores.iterrows():
        lines.append(
            f"  Status '{row['order_status']}': avg score {row['mean']:.2f} "
            f"({int(row['count'])} reviews)"
        )
    lines.append("")

    # FIX: chain filters properly with & instead of double-bracket indexing
    lines.append("SAMPLE NEGATIVE REVIEW MESSAGES (score = 1, non-empty)")
    lines.append("-" * 40)
    samples = (
        neg_reviews[
            (neg_reviews["review_score"] == 1) &
            (neg_reviews["review_comment_message"].str.len() > 20)
        ]["review_comment_message"]
        .dropna()
        .head(10)
    )
    for i, msg in enumerate(samples, 1):
        lines.append(f"  Sample {i}: {str(msg)[:200]}")

    return "\n".join(lines)


# Generate all reports
report_data = {
    "monthly_sales_summary":       report_monthly_sales(),
    "seller_performance_report":   report_seller_performance(),
    "delivery_performance_report": report_delivery_performance(),
    "customer_complaints_report":  report_customer_complaints(),
}
for name, content in report_data.items():
    path = OUT / f"{name}.txt"
    path.write_text(content, encoding="utf-8")
    print(f"   Saved: {path.name}  ({len(content):,} chars)")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — TEXT CHUNKING (BPE-aware, target 100-200 tokens, max 256)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Chunking reports (BPE-aware, target 100-200 tokens)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def count_bpe(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def chunk_text_bpe(text: str, doc_name: str,
                   min_tok: int = CHUNK_MIN,
                   max_tok: int = CHUNK_MAX) -> list[dict]:
    """
    Split document into chunks fitting within [min_tok, max_tok] BPE tokens.
    Strategy: split on double-newline paragraphs first, then single-newline
    lines if a paragraph is too large. Merge small paragraphs greedily.
    """
    chunks   = []
    chunk_id = 0

    def flush(buf: list[str]) -> None:
        nonlocal chunk_id
        merged = "\n\n".join(buf)
        toks   = count_bpe(merged)
        chunks.append({
            "chunk_id": f"{doc_name}_chunk_{chunk_id:03d}",
            "doc_name": doc_name,
            "text":     merged,
            "bpe_tokens": toks,
            "char_len": len(merged),
        })
        chunk_id += 1

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf, buf_tok = [], 0

    for para in paragraphs:
        pt = count_bpe(para)

        if pt > max_tok:
            # Para itself is too large.
            # FIX: save any small remnant in buf before we overwrite it below.
            # Without this, a small paragraph (< min_tok) sitting in buf gets
            # silently discarded when we assign `buf, buf_tok = lb, lt` at the end.
            if buf and buf_tok >= min_tok:
                flush(buf)
                buf, buf_tok = [], 0
            saved_buf = buf[:]      # may be a small remnant (buf_tok < min_tok)
            saved_tok = buf_tok

            lines  = [ln.strip() for ln in para.split("\n") if ln.strip()]
            lb, lt = [], 0
            for line in lines:
                llt = count_bpe(line)
                if lt + llt > max_tok and lt >= min_tok:
                    if saved_buf:   # prepend saved remnant to this first sub-chunk
                        lb   = saved_buf + lb
                        lt  += saved_tok
                        saved_buf, saved_tok = [], 0
                    flush(lb)
                    lb, lt = [line], llt
                else:
                    lb.append(line)
                    lt += llt
            if lb:
                if saved_buf:       # prepend any still-unplaced remnant
                    lb   = saved_buf + lb
                    lt  += saved_tok
                buf, buf_tok = lb, lt
        else:
            if buf_tok + pt > max_tok and buf_tok >= min_tok:
                flush(buf)
                buf, buf_tok = [para], pt
            else:
                buf.append(para)
                buf_tok += pt

    # Flush remainder
    if buf:
        if buf_tok < min_tok and chunks:
            last = chunks[-1]
            merged = last["text"] + "\n\n" + "\n\n".join(buf)
            chunks[-1] = {**last, "text": merged, "bpe_tokens": count_bpe(merged),
                          "char_len": len(merged)}
        else:
            flush(buf)

    return chunks


all_chunks: list[dict] = []
for doc_name, content in report_data.items():
    doc_chunks  = chunk_text_bpe(content, doc_name)
    all_chunks.extend(doc_chunks)
    bpe_counts  = [c["bpe_tokens"] for c in doc_chunks]
    over_limit  = sum(1 for t in bpe_counts if t > BPE_MAX)
    print(
        f"   {doc_name}: {len(doc_chunks)} chunks | "
        f"avg {sum(bpe_counts)/len(bpe_counts):.0f} BPE tokens | "
        f"range [{min(bpe_counts)}-{max(bpe_counts)}] | "
        f"over-256: {over_limit}"
    )

print(f"   Total chunks: {len(all_chunks)}")

chunk_store_path = OUT / "chunk_store.json"
with open(chunk_store_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
print(f"   Chunk store saved: {chunk_store_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Generating embeddings with all-MiniLM-L6-v2...")

model = SentenceTransformer(MODEL_NAME)
texts = [c["text"] for c in all_chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True,
)
embeddings = np.array(embeddings, dtype="float32")
print(f"   Embedding matrix shape: {embeddings.shape}")

# Verify no truncation warnings occurred (chunks should all be within 256 BPE)
over = sum(1 for c in all_chunks if c["bpe_tokens"] > BPE_MAX)
print(f"   Chunks exceeding model max ({BPE_MAX} BPE): {over}/{len(all_chunks)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD & SAVE FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Building FAISS index...")

dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # cosine similarity (embeddings are L2-normalized)
index.add(embeddings)
print(f"   FAISS index: {index.ntotal} vectors, dim={dim}")

faiss.write_index(index, str(OUT / "faiss_index.bin"))
with open(OUT / "faiss_metadata.pkl", "wb") as f:
    pickle.dump(all_chunks, f)
print(f"   Saved: faiss_index.bin + faiss_metadata.pkl")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — TEST VECTOR SEARCH
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] Testing vector search...")

def search(query: str, top_k: int = 3) -> list[dict]:
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)
    return [{**all_chunks[i], "score": float(s)} for s, i in zip(scores[0], indices[0])]

test_query = "Which region has worst delivery time?"
print(f"\n   Query: '{test_query}'")
print("   " + "-" * 60)
results = search(test_query, top_k=3)
for rank, r in enumerate(results, 1):
    print(f"\n   Rank {rank} | Score: {r['score']:.4f} | "
          f"Chunk: {r['chunk_id']} | BPE: {r['bpe_tokens']}")
    print(f"   Preview: {r['text'][:300].replace(chr(10), ' ')}...")

print("\n   Relevance check:")
relevant_terms = ["region", "delivery", "north", "northeast", "south",
                  "southeast", "midwest", "days", "late", "slow"]
for rank, r in enumerate(results, 1):
    found = [t for t in relevant_terms if t.lower() in r["text"].lower()]
    print(f"   Rank {rank}: Terms found -> {found}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — 20-QUESTION Q&A EVALUATION SET
# ══════════════════════════════════════════════════════════════════════════
print("\n[7/7] Building 20-question Q&A evaluation set...")

eval_questions = [
    # ── Simple Stats (5) ──────────────────────────────────────────────────
    {
        "id": "Q01", "category": "Simple Stats",
        "topic": "delivery",
        "question": "What is the total number of delivered orders in the dataset?",
        "expected_keywords": ["delivered", "orders", "total", "analysed"],
        "relevant_docs": ["delivery_performance_report"],
        "ground_truth_answer": "110,189 delivered orders were analysed.",
        "notes": "Answer is in the delivery report header paragraph.",
    },
    {
        "id": "Q02", "category": "Simple Stats",
        "topic": "sales",
        "question": "Which product category generates the highest total revenue?",
        "expected_keywords": ["health_beauty", "revenue", "rank 1"],
        "relevant_docs": ["monthly_sales_summary"],
        "ground_truth_answer": "health_beauty is ranked 1 by all-time revenue.",
        "notes": "Answer in all-time category revenue ranking, Rank 1.",
    },
    {
        "id": "Q03", "category": "Simple Stats",
        "topic": "operations",
        "question": "What percentage of customer reviews are negative (score 1 or 2)?",
        "expected_keywords": ["negative", "score 1", "score 2", "%"],
        "relevant_docs": ["customer_complaints_report"],
        "ground_truth_answer": "Lookup the 'Negative reviews (score 1-2)' line in the complaints report.",
        "notes": "Single numeric fact from review distribution table.",
    },
    {
        "id": "Q04", "category": "Simple Stats",
        "topic": "sellers",
        "question": "How many active sellers are there on the platform?",
        "expected_keywords": ["sellers", "total active", "3,"],
        "relevant_docs": ["seller_performance_report"],
        "ground_truth_answer": "Total active sellers figure from the seller performance report header.",
        "notes": "Header fact in seller performance report.",
    },
    {
        "id": "Q05", "category": "Simple Stats",
        "topic": "delivery",
        "question": "What is the overall average delivery time in days across all sellers?",
        "expected_keywords": ["avg delivery days", "overall"],
        "relevant_docs": ["seller_performance_report"],
        "ground_truth_answer": "Overall avg delivery days figure from the seller report summary.",
        "notes": "Numeric stat in seller report overall summary section.",
    },

    # ── Trend Analysis (5) ────────────────────────────────────────────────
    {
        "id": "Q06", "category": "Trend Analysis",
        "topic": "sales",
        "question": "How did total monthly revenue change from 2017 to 2018?",
        "expected_keywords": ["2017", "2018", "revenue", "month"],
        "relevant_docs": ["monthly_sales_summary"],
        "ground_truth_answer": "Compare monthly totals between 2017 and 2018 months in the OVERALL MONTHLY TOTALS section.",
        "notes": "Multi-month comparison -- identify growth direction.",
    },
    {
        "id": "Q07", "category": "Trend Analysis",
        "topic": "sales",
        "question": "Which product category appeared most frequently in the top-3 monthly revenue ranking?",
        "expected_keywords": ["category", "top 3", "month", "health_beauty", "watches_gifts"],
        "relevant_docs": ["monthly_sales_summary"],
        "ground_truth_answer": "Count category appearances across TOP 3 CATEGORIES PER MONTH sections.",
        "notes": "Requires scanning across multiple monthly top-3 tables.",
    },
    # FIX Q08: was 'seasonal late delivery pattern' -- unanswerable (no month x region data).
    # Replaced with a question directly answerable from the monthly totals chunks.
    {
        "id": "Q08", "category": "Trend Analysis",
        "topic": "sales",
        "question": "Which month had the highest total revenue in the entire dataset?",
        "expected_keywords": ["revenue", "month", "2018", "highest"],
        "relevant_docs": ["monthly_sales_summary"],
        "ground_truth_answer": "Find the month with the highest 'Revenue BRL' in OVERALL MONTHLY TOTALS.",
        "notes": "Direct lookup from monthly totals -- answerable from chunk store.",
    },
    # FIX Q09: was 'negative reviews improved over time' -- unanswerable (no time-series review data).
    # Replaced with a question answerable from the avg score by order status table.
    {
        "id": "Q09", "category": "Trend Analysis",
        "topic": "operations",
        "question": "How does the average review score differ between delivered and cancelled orders?",
        "expected_keywords": ["delivered", "cancelled", "review score", "avg score", "status"],
        "relevant_docs": ["customer_complaints_report"],
        "ground_truth_answer": "Compare avg score rows for 'delivered' vs 'cancelled' in AVERAGE REVIEW SCORE BY ORDER STATUS.",
        "notes": "Direct comparison from the order-status score breakdown table.",
    },
    {
        "id": "Q10", "category": "Trend Analysis",
        "topic": "sales",
        "question": "Which months in 2018 saw over 1 million BRL in revenue?",
        "expected_keywords": ["2018", "1,0", "revenue", "month"],
        "relevant_docs": ["monthly_sales_summary"],
        "ground_truth_answer": "Scan OVERALL MONTHLY TOTALS for 2018 months with Revenue > BRL 1,000,000.",
        "notes": "Threshold-based lookup from monthly totals.",
    },

    # ── Comparison (5) ────────────────────────────────────────────────────
    {
        "id": "Q11", "category": "Comparison",
        "topic": "delivery",
        "question": "Which region has the slowest average delivery time compared to others?",
        "expected_keywords": ["North", "slowest", "22.2 days", "delivery"],
        "relevant_docs": ["delivery_performance_report"],
        "ground_truth_answer": "North region -- avg 22.2 days (from REGIONAL SUMMARY sorted slowest first).",
        "notes": "Core retrieval test -- directly tested in vector search demo.",
    },
    {
        "id": "Q12", "category": "Comparison",
        "topic": "sellers",
        "question": "How do top-10 sellers compare to bottom-10 sellers in average review score and revenue?",
        "expected_keywords": ["top", "bottom", "seller", "rating", "revenue"],
        "relevant_docs": ["seller_performance_report"],
        "ground_truth_answer": "Read top-10 and bottom-10 seller tables and compare Rating and Revenue columns.",
        "notes": "Side-by-side comparison from top/bottom seller tables.",
    },
    {
        "id": "Q13", "category": "Comparison",
        "topic": "delivery",
        "question": "Which region has the highest percentage of late orders?",
        "expected_keywords": ["Northeast", "late", "12.6%", "MOST LATE ORDERS"],
        "relevant_docs": ["delivery_performance_report"],
        "ground_truth_answer": "Northeast region -- 12.6% of orders arrived after estimated date.",
        "notes": "Distinct from Q11: North=slowest absolute time, Northeast=highest late%.",
    },
    {
        "id": "Q14", "category": "Comparison",
        "topic": "sellers",
        "question": "Which seller state generates more revenue -- SP or MG?",
        "expected_keywords": ["SP", "MG", "revenue", "state"],
        "relevant_docs": ["seller_performance_report"],
        "ground_truth_answer": "SP generates more revenue than MG (from SELLER PERFORMANCE BY STATE table).",
        "notes": "State-level seller performance comparison.",
    },
    {
        "id": "Q15", "category": "Comparison",
        "topic": "delivery",
        "question": "How does average delivery time in the North region compare to the Southeast?",
        "expected_keywords": ["North", "Southeast", "22.2", "10.2", "days"],
        "relevant_docs": ["delivery_performance_report"],
        "ground_truth_answer": "North avg 22.2 days vs Southeast avg 10.2 days -- North is 2x slower.",
        "notes": "Direct region comparison from REGIONAL SUMMARY table.",
    },

    # ── Causal Reasoning (5) ─────────────────────────────────────────────
    # FIX Q16 expected_keywords: removed 'infrastructure', 'distance', 'remote'
    # (these words appear in NO chunk -- they are answer-generation keywords, not retrieval keywords).
    # expected_keywords should be terms found IN the retrieved chunks, not the LLM's ideal response.
    {
        "id": "Q16", "category": "Causal Reasoning",
        "topic": "delivery",
        "question": "Why might the North region have the slowest delivery performance?",
        "expected_keywords": ["North", "22.2 days", "slowest", "delivery", "avg"],
        "relevant_docs": ["delivery_performance_report"],
        "ground_truth_answer": "The data shows North has avg 22.2 days delivery. The LLM should infer geographic remoteness from the data pattern.",
        "notes": "expected_keywords are chunk-retrieval terms (in documents). Geographic reasoning is LLM inference.",
    },
    {
        "id": "Q17", "category": "Causal Reasoning",
        "topic": "operations",
        "question": "What is the most likely cause of 1-star reviews according to complaint themes?",
        "expected_keywords": ["Late Delivery", "Not Received", "1,770", "12.1%"],
        "relevant_docs": ["customer_complaints_report"],
        "ground_truth_answer": "Late Delivery / Not Received -- 1,770 mentions (12.1% of negative reviews), top-ranked theme.",
        "notes": "Direct lookup from complaint theme rankings.",
    },
    {
        "id": "Q18", "category": "Causal Reasoning",
        "topic": "sellers",
        "question": "Why do bottom-10 sellers have lower revenue -- is it poor ratings or delivery issues?",
        "expected_keywords": ["bottom", "seller", "revenue", "rating", "delivery"],
        "relevant_docs": ["seller_performance_report"],
        "ground_truth_answer": "Read bottom-10 table: compare Rating and Avg Delivery figures vs top-10 to determine dominant factor.",
        "notes": "Multi-factor reasoning across top and bottom seller tables.",
    },
    {
        "id": "Q19", "category": "Causal Reasoning",
        "topic": "operations",
        "question": "How might slow deliveries in the North region contribute to negative reviews?",
        "expected_keywords": ["North", "Late Delivery", "negative", "22.2", "delivery"],
        "relevant_docs": ["delivery_performance_report", "customer_complaints_report"],
        "ground_truth_answer": "North has 22.2 avg delivery days; Late Delivery is the #1 complaint (21%). Cross-doc causal link.",
        "notes": "Cross-document causal chain: delivery delay -> Late Delivery complaint -> low score.",
    },
    {
        "id": "Q20", "category": "Causal Reasoning",
        "topic": "sellers",
        "question": "If a seller operates in a state with slow delivery, how does that correlate with their review score?",
        "expected_keywords": ["state", "delivery", "avg delivery", "rating", "seller"],
        "relevant_docs": ["seller_performance_report", "delivery_performance_report"],
        "ground_truth_answer": "Compare SELLER PERFORMANCE BY STATE table: states with higher avg delivery days vs their avg rating.",
        "notes": "expected_keywords corrected: removed 'infrastructure' (not in chunks). Hypothesis validated via seller-state table.",
    },
]

categories = {}
for q in eval_questions:
    categories[q["category"]] = categories.get(q["category"], 0) + 1

print(f"\n   Total questions: {len(eval_questions)}")
for cat, count in categories.items():
    print(f"   {cat}: {count} questions")

with open(OUT / "eval_set_20q.json", "w", encoding="utf-8") as f:
    json.dump(eval_questions, f, ensure_ascii=False, indent=2)

with open(OUT / "eval_set_20q.txt", "w", encoding="utf-8") as f:
    f.write("20-QUESTION Q&A EVALUATION SET\nRAG Agent - Day 1 (v3 fully corrected)\n" + "=" * 60 + "\n\n")
    for q in eval_questions:
        f.write(f"[{q['id']}] [{q['category']}] [Topic: {q.get('topic','?')}]\n")
        f.write(f"Q: {q['question']}\n")
        f.write(f"   Expected keywords (in chunks): {', '.join(q['expected_keywords'])}\n")
        f.write(f"   Relevant docs: {', '.join(q['relevant_docs'])}\n")
        f.write(f"   Ground truth:  {q.get('ground_truth_answer', 'N/A')}\n")
        f.write(f"   Notes:         {q['notes']}\n\n")

print(f"   Eval set saved.")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DAY 1 OUTPUT SUMMARY  (v2 corrected)")
print("=" * 70)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<42} {f.stat().st_size/1024:>8.1f} KB")

print(f"\n  FAISS index:   {index.ntotal} vectors, dim={dim}")
print(f"  Chunk store:   {len(all_chunks)} chunks")
print(f"  BPE range:     [{min(c['bpe_tokens'] for c in all_chunks)}"
      f"-{max(c['bpe_tokens'] for c in all_chunks)}] tokens")
print(f"  Chunks over {BPE_MAX} BPE: {sum(1 for c in all_chunks if c['bpe_tokens']>BPE_MAX)}")
print(f"  Eval set:      {len(eval_questions)} questions across {len(categories)} categories")
print("\nAll Day 1 artifacts saved to:", OUT)
print("=" * 70)
