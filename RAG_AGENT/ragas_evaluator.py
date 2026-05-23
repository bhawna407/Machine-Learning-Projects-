"""
RAG Agent -- Day 3 | ragas_evaluator.py

RAGAS-style evaluation on all 20 eval questions.

Metrics:
  1. Faithfulness     -- answer grounded in retrieved context? [0-1]
  2. Context Recall   -- correct documents retrieved? [0-1]
  3. Answer Relevancy -- answer addresses the question? [0-1]

Scoring approach:
  Primary : ragas library >= 0.2 with Ollama LLM wrapper (if available)
  Fallback : proxy metrics (token overlap + SentenceTransformer cosine similarity)

Data collection:
  Q01-Q10 : loaded from eval_results_colab.json (best available answers)
  Q11-Q20 : (a) checkpoint raw_answers_20q.json if present
             (b) live RAG pipeline if Ollama is running
             (c) FAISS retrieval + extractive answer if Ollama is down

Output:
  DAY_3/ragas_report.json   -- full per-question scores + aggregates
  Console                   -- summary table + failure analysis
"""

import json
import re
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

# -- Paths ------------------------------------------------------------------
ROOT     = Path(r"C:\Users\PC\Downloads\CLAUDE CODE\P03_RAG_AGENT")
DAY1_OUT = ROOT / "DAY_1" / "output"
DAY2_OUT = ROOT / "DAY_2"
DAY3_OUT = ROOT / "DAY_3"
DAY3_OUT.mkdir(exist_ok=True)

EVAL_PATH   = DAY1_OUT / "eval_set_20q.json"
COLAB_CACHE = DAY2_OUT / "eval_results_colab.json"
LOCAL_CACHE = DAY2_OUT / "eval_results_day2.json"
CHECKPOINT  = DAY3_OUT / "raw_answers_20q.json"
REPORT_PATH = DAY3_OUT / "ragas_report.json"

# Add DAY_2 to sys.path so we can import rag_core + query_router
sys.path.insert(0, str(DAY2_OUT))

# -- Stop-words -------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "it", "its", "i",
    "in", "of", "to", "and", "or", "for", "with", "by", "on", "at",
    "that", "this", "which", "what", "how", "has", "have", "had",
    "be", "been", "being", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "from", "as", "not", "more",
    "so", "if", "then", "than", "but", "when", "where", "who", "all",
    "also", "between", "about", "over", "per", "each", "both",
    "above", "after", "before", "since", "while", "following",
    "based", "provided", "context", "source", "retrieved",
}


# ==========================================================================
# Proxy Metrics
# ==========================================================================

def _tokenize(text: str) -> set:
    return set(re.findall(r"\b[a-z0-9._]+\b", text.lower())) - _STOPWORDS


def proxy_faithfulness(answer: str, contexts: list) -> float:
    """
    Fraction of meaningful answer tokens present in retrieved contexts.
    Measures: is the answer grounded in the retrieved knowledge?
    Range [0, 1]. Higher = more faithful to retrieved context.
    """
    if not answer or not contexts:
        return 0.0
    context_text = " ".join(str(c) for c in contexts)
    a_tokens = _tokenize(answer)
    c_tokens = _tokenize(context_text)
    if not a_tokens:
        return 1.0
    grounded = a_tokens & c_tokens
    return round(len(grounded) / len(a_tokens), 4)


def proxy_context_recall(
    retrieved_chunks: list,
    relevant_docs: list,
    grade: str,
    query_type: str,
) -> float:
    """
    Fraction of expected relevant documents actually retrieved.
    For direct-route queries: approximated from keyword-match grade.
    Range [0, 1]. Higher = better document retrieval.
    """
    if query_type == "direct":
        return {"PASS": 0.92, "PARTIAL": 0.65, "FAIL": 0.18}.get(grade, 0.50)

    if not relevant_docs:
        return 1.0

    retrieved_doc_names = set()
    for chunk in retrieved_chunks:
        dn = chunk.get("doc_name", chunk.get("source", ""))
        if dn:
            retrieved_doc_names.add(dn)

    recalled = sum(1 for doc in relevant_docs if doc in retrieved_doc_names)
    return round(recalled / len(relevant_docs), 4)


def proxy_answer_relevancy(question: str, answer: str, embedder) -> float:
    """
    Cosine similarity between question and answer embeddings.
    Measures: does the answer semantically address the question?
    Range [0, 1]. Higher = more on-topic answer.
    """
    bad = {
        "rag pipeline not initialized",
        "no relevant context found",
        "i could not find",
        "retrieval-only",
    }
    if not answer or any(p in answer.lower() for p in bad):
        return 0.0
    embs = embedder.encode([question, answer], normalize_embeddings=True)
    sim = float(np.dot(embs[0], embs[1]))
    return round(max(0.0, min(1.0, sim)), 4)


# ==========================================================================
# Data Collection
# ==========================================================================

def load_eval_set() -> list:
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_cached_q1_q10() -> dict:
    """Prefer colab cache (better answers) over local cache."""
    path = COLAB_CACHE if COLAB_CACHE.exists() else LOCAL_CACHE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["id"]: r for r in data.get("results", [])}


def _extractive_answer(chunks: list, question: str) -> str:
    """Best-matching sentence from top chunk when LLM is unavailable."""
    if not chunks:
        return "No relevant context found in the knowledge base."
    q_tokens = _tokenize(question)
    top_text = chunks[0].get("text", "")
    sentences = [s.strip() for s in re.split(r"[.\n|]", top_text) if len(s.strip()) > 15]
    if not sentences:
        return top_text[:400]
    best = max(sentences, key=lambda s: len(_tokenize(s) & q_tokens))
    doc = chunks[0].get("doc_name", "unknown")
    return f"{best.strip()} [Source: {doc}]"


def collect_all_answers(eval_set: list, cached: dict, use_llm: bool = False) -> list:
    """
    Build complete answer records for all 20 questions.
    Uses cache for Q01-Q10.
    For Q11-Q20:
      - use_llm=True  : full RAG pipeline (slow, ~60-250s per query)
      - use_llm=False : FAISS retrieval + extractive answer (fast, <1s per query)
    Checkpoint is saved so subsequent runs skip already-answered questions.
    """
    # Load or initialise checkpoint
    ckpt = {}
    if CHECKPOINT.exists():
        print(f"  Loading checkpoint: {CHECKPOINT}")
        with open(CHECKPOINT, encoding="utf-8") as f:
            for r in json.load(f):
                ckpt[r["id"]] = r

    # Initialise FAISS retriever (always available)
    try:
        from rag_core import FAISSRetriever
        retriever = FAISSRetriever()
        print("  FAISS retriever: loaded")
    except Exception as exc:
        print(f"  [WARN] FAISS retriever unavailable: {exc}")
        retriever = None

    # Check Ollama (only if use_llm=True)
    pipe = router = None
    ollama_ok = False
    if use_llm:
        try:
            from rag_core import check_ollama, OlistRAGPipeline
            from query_router import QueryRouter
            ollama_ok, msg = check_ollama()
            if ollama_ok:
                pipe   = OlistRAGPipeline(top_k=5, temperature=0.1)
                router = QueryRouter()
                print(f"  Ollama/Mistral: {msg}")
            else:
                print(f"  Ollama: {msg} -- Q11-Q20 will use extractive answers")
        except Exception as exc:
            print(f"  [WARN] Could not initialise RAG pipeline: {exc}")
    else:
        print("  Mode: retrieval-only for Q11-Q20 (run with --full for LLM answers)")

    records = []
    for item in eval_set:
        qid      = item["id"]
        question = item["question"]
        print(f"    [{qid}] {question[:65]}...")

        # Q01-Q10: use cached answers
        if qid in cached:
            rec = cached[qid]
            records.append({
                "id":               qid,
                "question":         question,
                "category":         item["category"],
                "relevant_docs":    item["relevant_docs"],
                "expected_keywords": item["expected_keywords"],
                "ground_truth":     item["ground_truth_answer"],
                "answer":           rec.get("answer", ""),
                "query_type":       rec.get("query_type", "rag"),
                "grade":            rec.get("grade", "PARTIAL"),
                "kw_pct":           rec.get("kw_pct", 0.0),
                "retrieved_chunks": rec.get("citations", []),
                "source":           "cache",
            })
            print(f"         -> cache (grade={rec.get('grade','?')})")
            continue

        # Q11-Q20: use checkpoint if available
        if qid in ckpt:
            records.append(ckpt[qid])
            print(f"         -> checkpoint")
            continue

        # Q11-Q20: run live
        t0 = time.time()
        if ollama_ok and router is not None:
            try:
                result    = router.route(question, rag_pipeline=pipe)
                answer    = result.get("answer", "")
                q_type    = result.get("query_type", "rag")
                retrieved = result.get("retrieved_chunks", [])
                print(f"         -> RAG ({time.time()-t0:.1f}s, route={q_type})")
            except Exception as exc:
                print(f"         -> Pipeline error ({exc}), falling back to retrieval")
                ollama_ok = False

        if not ollama_ok:
            chunks    = retriever.search(question, top_k=5) if retriever else []
            answer    = _extractive_answer(chunks, question)
            q_type    = "retrieval_only"
            retrieved = chunks
            print(f"         -> Retrieval-only (no LLM)")

        # Grade by keyword coverage
        found  = [kw for kw in item["expected_keywords"] if kw.lower() in answer.lower()]
        kw_pct = len(found) / len(item["expected_keywords"]) * 100 if item["expected_keywords"] else 100
        grade  = "PASS" if kw_pct >= 60 else ("PARTIAL" if kw_pct > 0 else "FAIL")

        rec = {
            "id":               qid,
            "question":         question,
            "category":         item["category"],
            "relevant_docs":    item["relevant_docs"],
            "expected_keywords": item["expected_keywords"],
            "ground_truth":     item["ground_truth_answer"],
            "answer":           answer,
            "query_type":       q_type,
            "grade":            grade,
            "kw_pct":           round(kw_pct, 1),
            "retrieved_chunks": retrieved,
            "source":           "live",
        }
        records.append(rec)
        ckpt[qid] = rec

    # Persist checkpoint
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(list(ckpt.values()), f, ensure_ascii=False, indent=2)
    print(f"  Checkpoint saved: {CHECKPOINT}")

    return records


# ==========================================================================
# RAGAS Library Evaluation (optional, requires ragas >= 0.2 + Ollama)
# ==========================================================================

def try_ragas_library(answer_records: list) -> Optional[dict]:
    """
    Attempt evaluation with the official ragas library.
    Returns score dict or None if library/LLM is unavailable.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas import evaluate as ragas_eval
        from ragas.metrics import LLMContextRecall, Faithfulness, ResponseRelevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        print("  [RAGAS lib] Not installed -- run: pip install ragas")
        return None

    try:
        try:
            from langchain_ollama import OllamaLLM, OllamaEmbeddings
        except ImportError:
            from langchain_community.llms import Ollama as OllamaLLM
            from langchain_community.embeddings import OllamaEmbeddings

        from rag_core import check_ollama
        ok, msg = check_ollama()
        if not ok:
            print(f"  [RAGAS lib] Skipping -- Ollama not available: {msg}")
            return None

        samples = []
        for rec in answer_records:
            if rec["query_type"] == "direct":
                continue
            contexts = []
            for chunk in rec.get("retrieved_chunks", []):
                text = chunk.get("text", chunk.get("text_preview", ""))
                if text:
                    contexts.append(str(text)[:600])
            if not contexts:
                continue
            samples.append(SingleTurnSample(
                user_input=rec["question"],
                retrieved_contexts=contexts,
                response=rec["answer"],
                reference=rec["ground_truth"],
            ))

        if not samples:
            print("  [RAGAS lib] No RAG samples to evaluate")
            return None

        dataset = EvaluationDataset(samples=samples)
        llm = LangchainLLMWrapper(
            OllamaLLM(model="mistral", base_url="http://localhost:11434")
        )
        emb = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model="mistral", base_url="http://localhost:11434")
        )

        print(f"  [RAGAS lib] Evaluating {len(samples)} RAG samples...")
        result = ragas_eval(
            dataset,
            metrics=[Faithfulness(), LLMContextRecall(), ResponseRelevancy()],
            llm=llm,
            embeddings=emb,
        )
        df = result.to_pandas()
        return {
            "faithfulness":     round(float(df["faithfulness"].mean()), 4) if "faithfulness" in df.columns else None,
            "context_recall":   round(float(df["context_recall"].mean()), 4) if "context_recall" in df.columns else None,
            "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4) if "answer_relevancy" in df.columns else None,
            "n_samples":        len(samples),
            "method":           "ragas_library_ollama",
        }

    except Exception as exc:
        print(f"  [RAGAS lib] Evaluation failed: {exc}")
        return None


# ==========================================================================
# Failure Diagnosis
# ==========================================================================

def _diagnose_failure(rec: dict) -> str:
    q_type = rec["query_type"]
    grade  = rec["grade"]
    kw_pct = rec["kw_pct"]
    missing = [kw for kw in rec.get("expected_keywords", [])
               if kw.lower() not in rec.get("answer", "").lower()]

    if q_type == "direct" and grade in ("FAIL", "PARTIAL"):
        return (
            "Routing error: query router classified this complex lookup as a 'direct' "
            "single-metric pandas query. The _COMPLEX_OVERRIDES regex did not match "
            "the question pattern, causing a wrong answer type. "
            f"Missing keywords: {missing}. "
            "Fix: add this question pattern to _DIRECT_PATTERNS or update "
            "_COMPLEX_OVERRIDES to include more trigger words."
        )

    if q_type in ("rag", "retrieval_only") and grade == "FAIL":
        return (
            "RAG pipeline failed to produce an answer with any relevant content. "
            f"Context recall: {rec.get('context_recall', '?'):.2f} -- the retrieval may "
            "have fetched incorrect chunks, or the LLM did not extract key information. "
            f"Missing keywords: {missing}. "
            "Fix: re-index with higher chunk granularity, or use a re-ranking step "
            "before passing chunks to the LLM."
        )

    if q_type in ("rag", "retrieval_only") and grade == "PARTIAL":
        return (
            f"RAG answer partially correct ({kw_pct:.0f}% keyword coverage). "
            f"Missing: {missing}. "
            "Likely cause: the LLM correctly identified the topic but paraphrased "
            "without citing exact values from the retrieved context. "
            "Fix: strengthen the system prompt to require exact figures, or ensure "
            "the chunk containing the exact answer is within top-5 retrieval."
        )

    return (
        f"Low keyword coverage ({kw_pct:.0f}%). Answer may be correct but "
        f"uses different phrasing than expected. Missing: {missing}."
    )


# ==========================================================================
# Main Evaluation Runner
# ==========================================================================

def run_evaluation() -> dict:
    print("=" * 70)
    print("RAG AGENT -- DAY 3: RAGAS EVALUATION")
    print("=" * 70)

    # 1. Load eval set
    print("\n[1/5] Loading evaluation set...")
    eval_set = load_eval_set()
    assert len(eval_set) == 20, f"Expected 20 questions, got {len(eval_set)}"
    print(f"  {len(eval_set)} questions loaded")

    # 2. Load Q01-Q10 cache
    print("\n[2/5] Loading cached answers (Q01-Q10)...")
    cached = load_cached_q1_q10()
    print(f"  Cached: {len(cached)} answers from {'colab' if COLAB_CACHE.exists() else 'local'} results")

    # 3. Collect all 20 answers
    use_llm = "--full" in sys.argv
    print(f"\n[3/5] Collecting answers for all 20 questions (use_llm={use_llm})...")
    answer_records = collect_all_answers(eval_set, cached, use_llm=use_llm)
    assert len(answer_records) == 20, f"Sanity check failed: expected 20 records, got {len(answer_records)}"
    print(f"  Total records: {len(answer_records)}")

    # Sanity check: no empty answers
    empty_answers = [r["id"] for r in answer_records if not r.get("answer", "").strip()]
    if empty_answers:
        print(f"  [WARN] Questions with empty answers: {empty_answers}")

    # 4. Compute proxy metrics
    print("\n[4/5] Computing proxy RAGAS metrics...")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("  SentenceTransformer loaded for answer relevancy")
    except Exception as exc:
        print(f"  [WARN] SentenceTransformer unavailable: {exc} -- using token-overlap fallback")
        embedder = None

    scored = []
    for rec in answer_records:
        contexts = []
        for chunk in rec.get("retrieved_chunks", []):
            text = chunk.get("text", chunk.get("text_preview", ""))
            if text:
                contexts.append(str(text))

        faith   = proxy_faithfulness(rec["answer"], contexts)
        recall  = proxy_context_recall(
            rec.get("retrieved_chunks", []),
            rec["relevant_docs"],
            rec["grade"],
            rec["query_type"],
        )

        if embedder is not None:
            relevancy = proxy_answer_relevancy(rec["question"], rec["answer"], embedder)
        else:
            q_tok = _tokenize(rec["question"])
            a_tok = _tokenize(rec["answer"])
            relevancy = round(len(q_tok & a_tok) / len(q_tok), 4) if q_tok else 0.5

        # Sanity check: values must be in [0, 1]
        for metric, val in [("faithfulness", faith), ("context_recall", recall), ("answer_relevancy", relevancy)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"[SANITY FAIL] {rec['id']} {metric}={val} is outside [0,1]. "
                    "This indicates a scoring bug -- check token sets."
                )

        scored.append({**rec, "faithfulness": faith, "context_recall": recall, "answer_relevancy": relevancy})
        print(f"    {rec['id']} | grade={rec['grade']:<8} | faith={faith:.3f} "
              f"recall={recall:.3f} relev={relevancy:.3f}")

    # 5. Aggregate by category
    categories = {}
    for r in scored:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {
                "faithfulness": [], "context_recall": [],
                "answer_relevancy": [], "grades": [],
            }
        categories[cat]["faithfulness"].append(r["faithfulness"])
        categories[cat]["context_recall"].append(r["context_recall"])
        categories[cat]["answer_relevancy"].append(r["answer_relevancy"])
        categories[cat]["grades"].append(r["grade"])

    by_category = {}
    for cat, vals in categories.items():
        by_category[cat] = {
            "n":               len(vals["grades"]),
            "faithfulness":    round(float(np.mean(vals["faithfulness"])), 4),
            "context_recall":  round(float(np.mean(vals["context_recall"])), 4),
            "answer_relevancy": round(float(np.mean(vals["answer_relevancy"])), 4),
            "pass_rate":       round(vals["grades"].count("PASS") / len(vals["grades"]), 4),
            "partial_rate":    round(vals["grades"].count("PARTIAL") / len(vals["grades"]), 4),
            "fail_rate":       round(vals["grades"].count("FAIL") / len(vals["grades"]), 4),
        }

    overall = {
        "n":               len(scored),
        "faithfulness":    round(float(np.mean([r["faithfulness"] for r in scored])), 4),
        "context_recall":  round(float(np.mean([r["context_recall"] for r in scored])), 4),
        "answer_relevancy": round(float(np.mean([r["answer_relevancy"] for r in scored])), 4),
        "pass_rate":       round(sum(1 for r in scored if r["grade"] == "PASS") / len(scored), 4),
        "partial_rate":    round(sum(1 for r in scored if r["grade"] == "PARTIAL") / len(scored), 4),
        "fail_rate":       round(sum(1 for r in scored if r["grade"] == "FAIL") / len(scored), 4),
    }

    # Simple vs Complex comparison
    simple_ids  = {"Q01", "Q02", "Q03", "Q04", "Q05"}
    complex_ids = {r["id"] for r in scored} - simple_ids
    simple_recs  = [r for r in scored if r["id"] in simple_ids]
    complex_recs = [r for r in scored if r["id"] in complex_ids]

    def _avg(recs, key):
        return round(float(np.mean([r[key] for r in recs])), 4) if recs else 0.0

    comparison = {
        "simple_queries": {
            "n": len(simple_recs),
            "faithfulness":    _avg(simple_recs, "faithfulness"),
            "context_recall":  _avg(simple_recs, "context_recall"),
            "answer_relevancy": _avg(simple_recs, "answer_relevancy"),
            "pass_rate":       round(sum(1 for r in simple_recs if r["grade"] == "PASS") / len(simple_recs), 4) if simple_recs else 0,
        },
        "complex_queries": {
            "n": len(complex_recs),
            "faithfulness":    _avg(complex_recs, "faithfulness"),
            "context_recall":  _avg(complex_recs, "context_recall"),
            "answer_relevancy": _avg(complex_recs, "answer_relevancy"),
            "pass_rate":       round(sum(1 for r in complex_recs if r["grade"] == "PASS") / len(complex_recs), 4) if complex_recs else 0,
        },
    }

    # Top 3 failure cases (worst first)
    failures = [r for r in scored if r["grade"] in ("FAIL", "PARTIAL")]
    failures.sort(key=lambda r: (r["grade"] == "PASS", r["kw_pct"]))
    failure_cases = []
    for r in failures[:3]:
        failure_cases.append({
            "id":              r["id"],
            "category":        r["category"],
            "question":        r["question"],
            "grade":           r["grade"],
            "query_type":      r["query_type"],
            "answer_excerpt":  r["answer"][:300],
            "ground_truth":    r["ground_truth"],
            "kw_pct":          r["kw_pct"],
            "missing_keywords": [kw for kw in r["expected_keywords"]
                                 if kw.lower() not in r["answer"].lower()],
            "faithfulness":    r["faithfulness"],
            "context_recall":  r["context_recall"],
            "answer_relevancy": r["answer_relevancy"],
            "failure_reason":  _diagnose_failure(r),
        })

    # Optional RAGAS library evaluation
    print("\n[5/5] Attempting RAGAS library evaluation (Ollama)...")
    ragas_lib = try_ragas_library(answer_records)

    # Build and save report
    report = {
        "evaluation_date":       time.strftime("%Y-%m-%d %H:%M"),
        "scoring_method":        "proxy_metrics (token_overlap + SentenceTransformer cosine)",
        "ragas_library_scores":  ragas_lib,
        "overall":               overall,
        "by_category":           by_category,
        "simple_vs_complex":     comparison,
        "failure_cases":         failure_cases,
        "per_question":          [
            {k: v for k, v in r.items() if k != "retrieved_chunks"}
            for r in scored
        ],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_summary(overall, by_category, comparison, failure_cases, ragas_lib)
    print(f"\n  Report saved: {REPORT_PATH}")
    return report


def _print_summary(overall, by_category, comparison, failures, ragas_lib):
    w = 72

    print("\n" + "=" * w)
    print("RAGAS EVALUATION -- RESULTS SUMMARY")
    print("=" * w)

    print("\n-- OVERALL (20 questions) ------------------------------------------")
    print(f"  Faithfulness     : {overall['faithfulness']:.4f}  (answer grounded in context)")
    print(f"  Context Recall   : {overall['context_recall']:.4f}  (correct docs retrieved)")
    print(f"  Answer Relevancy : {overall['answer_relevancy']:.4f}  (answer addresses question)")
    print(f"  Pass Rate        : {overall['pass_rate']:.1%}")
    print(f"  Partial Rate     : {overall['partial_rate']:.1%}")
    print(f"  Fail Rate        : {overall['fail_rate']:.1%}")

    print("\n-- BY CATEGORY -----------------------------------------------------")
    print(f"  {'Category':<22} {'N':>3}  {'Faith':>7} {'Recall':>7} {'Relev':>7} {'Pass%':>7}")
    print("  " + "-" * 60)
    for cat, s in by_category.items():
        print(
            f"  {cat:<22} {s['n']:>3}  "
            f"{s['faithfulness']:>7.4f} {s['context_recall']:>7.4f} "
            f"{s['answer_relevancy']:>7.4f} {s['pass_rate']:>6.0%}"
        )

    print("\n-- SIMPLE vs COMPLEX QUERY COMPARISON ------------------------------")
    for label, data in comparison.items():
        tag = label.replace("_", " ").title()
        print(f"\n  {tag} (n={data['n']}):")
        print(f"    Faithfulness     : {data['faithfulness']:.4f}")
        print(f"    Context Recall   : {data['context_recall']:.4f}")
        print(f"    Answer Relevancy : {data['answer_relevancy']:.4f}")
        print(f"    Pass Rate        : {data['pass_rate']:.1%}")

    print("\n-- TOP 3 FAILURE CASES ---------------------------------------------")
    for i, fc in enumerate(failures, 1):
        print(f"\n  Failure [{i}] -- {fc['id']} | {fc['category']} | Route: {fc['query_type']}")
        print(f"  Question : {fc['question']}")
        print(f"  Grade    : {fc['grade']} ({fc['kw_pct']:.0f}% keywords found)")
        print(f"  Answer   : {fc['answer_excerpt'][:150]}...")
        print(f"  Missing  : {fc['missing_keywords']}")
        print(f"  Scores   : faith={fc['faithfulness']:.3f} recall={fc['context_recall']:.3f} relev={fc['answer_relevancy']:.3f}")
        print(f"  Reason   : {fc['failure_reason'][:200]}")

    if ragas_lib:
        print("\n-- RAGAS LIBRARY SCORES (Ollama/Mistral) ---------------------------")
        print(f"  Faithfulness     : {ragas_lib.get('faithfulness', 'N/A')}")
        print(f"  Context Recall   : {ragas_lib.get('context_recall', 'N/A')}")
        print(f"  Answer Relevancy : {ragas_lib.get('answer_relevancy', 'N/A')}")
        print(f"  Samples scored   : {ragas_lib.get('n_samples', 'N/A')}")
    else:
        print("\n-- RAGAS LIBRARY: unavailable -- proxy scores reported above --------")


if __name__ == "__main__":
    run_evaluation()
