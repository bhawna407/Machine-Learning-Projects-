"""
RAG Agent - Day 2 | rag_pipeline_day2.py
Main pipeline script:
  1. Verify Ollama / Mistral
  2. Initialize RAG pipeline + query router
  3. Evaluate on 10 questions from the Day-1 eval set
  4. Print accuracy report + save results JSON
"""

import json
import time
from pathlib import Path

from rag_core import OlistRAGPipeline, check_ollama
from query_router import QueryRouter

# ── Paths ──────────────────────────────────────────────────────────────────
DAY1_OUT = Path(r"C:\Users\PC\Downloads\CLAUDE CODE\P03_RAG_AGENT\DAY_1\output")
DAY2_OUT = Path(r"C:\Users\PC\Downloads\CLAUDE CODE\P03_RAG_AGENT\DAY_2")
DAY2_OUT.mkdir(exist_ok=True)

EVAL_PATH    = DAY1_OUT / "eval_set_20q.json"
RESULTS_PATH = DAY2_OUT / "eval_results_day2.json"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def score_answer(answer: str, expected_keywords: list[str]) -> tuple[str, list[str], list[str]]:
    """
    Check how many expected keywords appear in the answer.
    Returns (grade, found_keywords, missing_keywords).
    Grade: PASS (≥60%), PARTIAL (>0%), FAIL (0%)
    """
    answer_lower = answer.lower()
    found   = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    pct     = len(found) / len(expected_keywords) if expected_keywords else 1.0

    if pct >= 0.6:
        grade = "PASS"
    elif pct > 0.0:
        grade = "PARTIAL"
    else:
        grade = "FAIL"

    return grade, found, missing


def print_result(result: dict, grade: str, found: list, missing: list, elapsed: float) -> None:
    q = result.get("question", result.get("query", ""))
    print(f"\n  [{grade}] Q: {q[:80]}")
    print(f"         Route : {result['query_type'].upper()}")
    print(f"         Time  : {elapsed:.1f}s")
    print(f"         Answer: {result['answer'][:200]}")
    if result.get("citations"):
        cites = [c['chunk_id'] for c in result['citations']]
        print(f"         Cited : {cites}")
    print(f"         Found kw : {found}")
    if missing:
        print(f"         Missing  : {missing}")


# ══════════════════════════════════════════════════════════════════════════
# Main pipeline demo + evaluation
# ══════════════════════════════════════════════════════════════════════════

def run_demo_queries(pipe: OlistRAGPipeline, router: QueryRouter) -> None:
    """Run a few demo queries to show the pipeline working end-to-end."""
    demos = [
        "Which region has the slowest average delivery time compared to others?",
        "How many active sellers are there on the platform?",
        "What is the most likely cause of 1-star reviews according to complaint themes?",
        "How did total monthly revenue change from 2017 to 2018?",
    ]

    print("\n" + "=" * 70)
    print("DEMO QUERIES")
    print("=" * 70)

    for q in demos:
        print(f"\nQ: {q}")
        print("-" * 60)
        t0 = time.time()
        result = router.route(q, rag_pipeline=pipe)
        elapsed = time.time() - t0

        print(f"Route  : {result['query_type'].upper()} | {elapsed:.1f}s")
        print(f"Answer : {result['answer']}")
        if result.get("citations"):
            for c in result["citations"]:
                print(f"  [{c.get('source_num','?')}] {c['chunk_id']} | "
                      f"doc={c['doc_name']} | sim={c.get('similarity', 0):.4f}")


def run_eval(pipe: OlistRAGPipeline, router: QueryRouter) -> None:
    """Evaluate on first 10 questions from the Day-1 eval set."""

    if not EVAL_PATH.exists():
        print(f"[WARN] eval_set_20q.json not found at {EVAL_PATH}")
        return

    with open(EVAL_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    eval_questions = all_questions[:10]   # Q01-Q10

    print("\n" + "=" * 70)
    print("EVALUATION — 10 QUESTIONS FROM DAY-1 EVAL SET")
    print("=" * 70)
    print(f"{'ID':<5} {'Cat':<20} {'Route':<8} {'Grade':<8} "
          f"{'KW%':<6} {'Time':>6}   Question")
    print("-" * 70)

    records    = []
    counts     = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    by_route   = {"direct": [], "rag": []}
    total_time = 0.0

    for q_item in eval_questions:
        q_id       = q_item["id"]
        question   = q_item["question"]
        expected_kw = q_item["expected_keywords"]
        category   = q_item["category"]

        t0 = time.time()
        result = router.route(question, rag_pipeline=pipe)
        elapsed = time.time() - t0
        total_time += elapsed

        result["question"] = question

        grade, found, missing = score_answer(result["answer"], expected_kw)
        kw_pct = len(found) / len(expected_kw) * 100 if expected_kw else 100

        counts[grade] += 1
        by_route[result["query_type"]].append(grade)

        row = {
            "id":          q_id,
            "category":    category,
            "question":    question,
            "query_type":  result["query_type"],
            "answer":      result["answer"],
            "citations":   result.get("citations", []),
            "grade":       grade,
            "kw_pct":      round(kw_pct, 1),
            "found_kw":    found,
            "missing_kw":  missing,
            "elapsed_s":   round(elapsed, 2),
        }
        records.append(row)

        print(f"{q_id:<5} {category:<20} {result['query_type'].upper():<8} "
              f"{grade:<8} {kw_pct:5.1f}% {elapsed:5.1f}s   {question[:45]}...")

    # ── Summary ───────────────────────────────────────────────────────────
    total = len(eval_questions)
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total questions : {total}")
    print(f"  PASS            : {counts['PASS']}  ({counts['PASS']/total*100:.0f}%)")
    print(f"  PARTIAL         : {counts['PARTIAL']}  ({counts['PARTIAL']/total*100:.0f}%)")
    print(f"  FAIL            : {counts['FAIL']}  ({counts['FAIL']/total*100:.0f}%)")
    print(f"  Avg time/query  : {total_time/total:.1f}s")
    print(f"\n  Direct route    : {len(by_route['direct'])} queries "
          f"| PASS: {by_route['direct'].count('PASS')}")
    print(f"  RAG route       : {len(by_route['rag'])} queries "
          f"| PASS: {by_route['rag'].count('PASS')}")

    # ── Detailed per-question print ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for r in records:
        print_result(r, r["grade"], r["found_kw"], r["missing_kw"], r["elapsed_s"])

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "summary": {
            "total":   total,
            "pass":    counts["PASS"],
            "partial": counts["PARTIAL"],
            "fail":    counts["FAIL"],
            "pass_pct": round(counts["PASS"] / total * 100, 1),
            "avg_time_s": round(total_time / total, 2),
        },
        "results": records,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved: {RESULTS_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RAG AGENT - DAY 2 PIPELINE")
    print("=" * 70)

    # 1. Check Ollama
    print("\n[1/4] Checking Ollama / Mistral...")
    ok, msg = check_ollama()
    print(f"      Status: {msg}")

    if not ok:
        print("\n[ERROR] Mistral is not available. Pipeline cannot run LLM queries.")
        print("  Action: Make sure Ollama is running and run: ollama pull mistral")
        print("\n  Falling back to retrieval-only mode (no LLM generation).")
        from rag_core import FAISSRetriever
        retriever = FAISSRetriever()
        q = "Which region has the slowest average delivery time?"
        chunks = retriever.search(q, top_k=5)
        print(f"\n  Retrieval test: '{q}'")
        for i, c in enumerate(chunks, 1):
            print(f"    [{i}] {c['chunk_id']} | sim={c['similarity']:.4f}")
            print(f"        {c['text'][:150]}...")
        return

    # 2. Initialize pipeline
    print("\n[2/4] Initializing RAG pipeline...")
    pipe   = OlistRAGPipeline(top_k=5, temperature=0.1)
    router = QueryRouter()
    print("      Pipeline ready.")

    # 3. Demo queries
    print("\n[3/4] Running demo queries...")
    run_demo_queries(pipe, router)

    # 4. Evaluation
    print("\n[4/4] Running evaluation on 10 questions...")
    run_eval(pipe, router)

    print("\n" + "=" * 70)
    print("DAY 2 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Eval results : {RESULTS_PATH}")
    print(f"  Streamlit UI : run `streamlit run app.py` from DAY_2 directory")


if __name__ == "__main__":
    main()
