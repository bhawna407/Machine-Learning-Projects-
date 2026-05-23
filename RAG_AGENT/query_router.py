"""
RAG Agent - Day 2 | query_router.py
Query routing: simple numeric queries → direct pandas, complex → RAG pipeline
"""

import re
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\PC\Downloads\CLAUDE CODE\P03_RAG_AGENT\DAY_1") / "RAG AGENT DATA"


# ══════════════════════════════════════════════════════════════════════════
# Simple query patterns that can be answered with direct pandas
# ══════════════════════════════════════════════════════════════════════════
# Words that signal a comparison / analytical query — always route to RAG
# even if the query also contains a simple-metric keyword.
_COMPLEX_OVERRIDES = re.compile(
    r"\b(which|highest|lowest|most|least|best|worst|"
    r"differ|difference|compare|between|change|trend|"
    r"why|how did|what caused|frequently|category|region|state|month)\b",
    re.IGNORECASE,
)

_DIRECT_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, metric_key)
    (r"how many (?:total )?(?:delivered )?orders",              "total_delivered_orders"),
    (r"total (?:number of )?(?:delivered )?orders",             "total_delivered_orders"),
    (r"number of (?:delivered )?orders",                        "total_delivered_orders"),
    (r"how many (?:active )?sellers",                           "total_sellers"),
    (r"total (?:number of )?(?:active )?sellers",               "total_sellers"),
    (r"number of (?:active )?sellers",                          "total_sellers"),
    (r"how many customers",                                     "total_customers"),
    (r"total (?:number of )?customers",                         "total_customers"),
    (r"how many (?:total )?reviews",                            "total_reviews"),
    (r"total (?:number of )?reviews",                           "total_reviews"),
    (r"what (?:is|was) (?:the )?(?:overall )?(?:total )?revenue",  "total_revenue"),
    (r"total revenue",                                          "total_revenue"),
    (r"overall average delivery (?:time|days)",                 "avg_delivery_days"),
    (r"average delivery (?:time|days) (?:across all)?",        "avg_delivery_days"),
    (r"overall avg delivery",                                   "avg_delivery_days"),
    (r"what (?:is|was) (?:the )?average delivery",              "avg_delivery_days"),
    (r"(?:what|average) (?:is )?(?:the )?(?:overall )?avg(?:erage)? review score", "avg_review_score"),
    (r"average review score",                                   "avg_review_score"),
    (r"what percentage (?:of )?(?:customer )?reviews (?:are )?negative",
                                                                "pct_negative_reviews"),
    (r"(?:percentage|pct|percent) (?:of )?negative reviews",   "pct_negative_reviews"),
    (r"how many (?:unique )?products",                          "total_products"),
]

# ── Natural-language answer templates ─────────────────────────────────────
_ANSWER_TEMPLATES: dict[str, str] = {
    "total_delivered_orders": (
        "There are **{value:,}** delivered orders in the dataset."
    ),
    "total_sellers": (
        "There are **{value:,}** active sellers on the platform."
    ),
    "total_customers": (
        "There are **{value:,}** unique customers in the dataset."
    ),
    "total_reviews": (
        "There are **{value:,}** customer reviews in the dataset."
    ),
    "total_revenue": (
        "Total revenue from delivered orders is **BRL {value:,.2f}**."
    ),
    "avg_delivery_days": (
        "The overall average delivery time across all sellers is **{value:.1f} days**."
    ),
    "avg_review_score": (
        "The overall average customer review score is **{value:.2f} / 5.0**."
    ),
    "pct_negative_reviews": (
        "**{value:.1f}%** of customer reviews are negative (score 1 or 2)."
    ),
    "total_products": (
        "There are **{value:,}** unique products in the dataset."
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# Data loader (lazy, cached per-process)
# ══════════════════════════════════════════════════════════════════════════
_METRICS_CACHE: dict | None = None


def _load_metrics() -> dict:
    global _METRICS_CACHE
    if _METRICS_CACHE is not None:
        return _METRICS_CACHE

    orders    = pd.read_csv(DATA / "olist_orders_dataset.csv",
                            parse_dates=["order_purchase_timestamp",
                                         "order_delivered_customer_date"])
    items     = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    reviews   = pd.read_csv(DATA / "olist_order_reviews_dataset.csv")
    sellers   = pd.read_csv(DATA / "olist_sellers_dataset.csv")
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    products  = pd.read_csv(DATA / "olist_products_dataset.csv")

    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["days_to_deliver"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.days

    delivered_items = items[items["order_id"].isin(delivered["order_id"])]
    delivered_items = delivered_items.copy()
    delivered_items["revenue"] = delivered_items["price"] + delivered_items["freight_value"]

    total_revenue = delivered_items["revenue"].sum()

    delivered_with_days = delivered.dropna(subset=["days_to_deliver"])
    avg_delivery = delivered_with_days["days_to_deliver"].mean()

    rev_scores = reviews["review_score"]
    avg_review = rev_scores.mean()
    pct_neg    = (rev_scores <= 2).mean() * 100

    _METRICS_CACHE = {
        "total_delivered_orders": int(delivered["order_id"].nunique()),
        "total_sellers":          int(sellers["seller_id"].nunique()),
        "total_customers":        int(customers["customer_id"].nunique()),
        "total_reviews":          int(len(reviews)),
        "total_revenue":          float(total_revenue),
        "avg_delivery_days":      float(avg_delivery),
        "avg_review_score":       float(avg_review),
        "pct_negative_reviews":   float(pct_neg),
        "total_products":         int(products["product_id"].nunique()),
    }
    return _METRICS_CACHE


# ══════════════════════════════════════════════════════════════════════════
# Query Router
# ══════════════════════════════════════════════════════════════════════════

class QueryRouter:
    """
    Classifies each query as 'direct' (single-metric pandas lookup)
    or 'rag' (full retrieval pipeline).
    """

    def classify(self, query: str) -> str:
        """Return 'direct' or 'rag'."""
        q = query.lower().strip()
        # If query contains complex / comparative language, always use RAG
        if _COMPLEX_OVERRIDES.search(q):
            return "rag"
        # Otherwise check for simple single-metric patterns
        for pattern, _ in _DIRECT_PATTERNS:
            if re.search(pattern, q):
                return "direct"
        return "rag"

    def compute_direct(self, query: str) -> dict:
        """
        Compute the answer for a direct query using pandas.
        Returns a result dict compatible with the RAG pipeline output format.
        """
        q = query.lower().strip()
        metrics = _load_metrics()

        matched_key = None
        for pattern, metric_key in _DIRECT_PATTERNS:
            if re.search(pattern, q):
                matched_key = metric_key
                break

        if matched_key is None or matched_key not in metrics:
            return {
                "query":      query,
                "query_type": "direct",
                "answer":     "I could not find a direct computation for this query.",
                "citations":  [],
                "source":     "pandas",
            }

        value    = metrics[matched_key]
        template = _ANSWER_TEMPLATES.get(matched_key, "Value: {value}")
        answer   = template.format(value=value)

        return {
            "query":            query,
            "query_type":       "direct",
            "answer":           answer,
            "citations":        [{"doc_name": "pandas_direct", "chunk_id": matched_key,
                                  "similarity": 1.0,
                                  "text_preview": f"Computed directly: {matched_key} = {value}"}],
            "retrieved_chunks": [],
            "source":           "pandas",
            "metric_key":       matched_key,
            "raw_value":        value,
        }

    def route(self, query: str, rag_pipeline=None) -> dict:
        """
        Route query to the appropriate handler.
        If route='rag' and no rag_pipeline provided, returns an error dict.
        """
        route_type = self.classify(query)

        if route_type == "direct":
            return self.compute_direct(query)

        if rag_pipeline is None:
            return {
                "query":      query,
                "query_type": "rag",
                "answer":     "RAG pipeline not initialized.",
                "citations":  [],
                "retrieved_chunks": [],
            }

        return rag_pipeline.run(query)

    def get_all_metrics(self) -> dict:
        """Return all precomputed metrics (useful for UI stats panel)."""
        return _load_metrics()


# ── Self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = QueryRouter()

    test_cases = [
        ("How many delivered orders are in the dataset?",       "direct"),
        ("How many active sellers are on the platform?",         "direct"),
        ("What is total revenue from delivered orders?",         "direct"),
        ("What is the overall average delivery time?",           "direct"),
        ("What percentage of reviews are negative?",             "direct"),
        ("Which region has the slowest delivery time?",          "rag"),
        ("Why do bottom sellers have lower revenue?",            "rag"),
        ("How did monthly revenue change from 2017 to 2018?",   "rag"),
        ("Which category appears most in top-3 monthly rank?",  "rag"),
    ]

    print("QUERY ROUTER CLASSIFICATION TEST")
    print("=" * 60)
    all_pass = True
    for q, expected in test_cases:
        got = router.classify(q)
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {got:6s} | {q}")

    print(f"\nClassification: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    print("\nDIRECT COMPUTATION TEST")
    print("=" * 60)
    metrics = router.get_all_metrics()
    for key, val in metrics.items():
        print(f"  {key:<28}: {val}")
