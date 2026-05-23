"""
RAG Agent - Day 2 | rag_core.py
Core pipeline: embedder → FAISS retriever → prompt builder → Mistral-7B → citations
"""

import json
import pickle
import re
import numpy as np
import faiss
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
try:
    from langchain_ollama import OllamaLLM as Ollama          # preferred (langchain-ollama)
except ImportError:
    from langchain_community.llms import Ollama               # fallback
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Paths ──────────────────────────────────────────────────────────────────
DAY1_OUT = Path(r"C:\Users\PC\Downloads\CLAUDE CODE\P03_RAG_AGENT\DAY_1\output")

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
OLLAMA_BASE  = "http://localhost:11434"

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert data analyst for Olist, a Brazilian e-commerce marketplace.
Answer the user's question using ONLY the information from the retrieved context below.
Be specific and include exact numbers when available.
If the context does not contain enough information, say so clearly.
At the end of your answer, on a separate line write:
SOURCES: [list the source numbers you used, e.g. [1], [3]]"""

RAG_PROMPT_TEMPLATE = """{system}

=== RETRIEVED CONTEXT ===
{context}
=== END CONTEXT ===

Question: {question}

Answer:"""


# ══════════════════════════════════════════════════════════════════════════
# FAISS Retriever
# ══════════════════════════════════════════════════════════════════════════

class FAISSRetriever:
    """Loads the Day-1 FAISS index and chunk metadata; retrieves top-K chunks."""

    def __init__(self):
        index_path = DAY1_OUT / "faiss_index.bin"
        meta_path  = DAY1_OUT / "faiss_metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Day 1 artifacts not found in {DAY1_OUT}. "
                "Run DAY_1/rag_pipeline.py first."
            )

        self.index  = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)

        self.embedder = SentenceTransformer(EMBED_MODEL)
        print(f"[FAISSRetriever] Loaded {self.index.ntotal} vectors | "
              f"{len(self.chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunks most similar to query, with similarity scores."""
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["similarity"] = float(score)
            results.append(chunk)

        return results


# ══════════════════════════════════════════════════════════════════════════
# Prompt Builder
# ══════════════════════════════════════════════════════════════════════════

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        doc   = chunk.get("doc_name", "unknown")
        cid   = chunk.get("chunk_id", f"chunk_{i}")
        score = chunk.get("similarity", 0.0)
        text  = chunk.get("text", "")
        parts.append(
            f"[SOURCE {i}] | doc: {doc} | chunk: {cid} | relevance: {score:.4f}\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the full RAG prompt."""
    context = build_context(chunks)
    return RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )


# ══════════════════════════════════════════════════════════════════════════
# LLM (Mistral-7B via Ollama)
# ══════════════════════════════════════════════════════════════════════════

def make_llm(temperature: float = 0.1) -> Any:
    """Return a LangChain Ollama LLM instance."""
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE,
        temperature=temperature,
    )


def check_ollama() -> tuple[bool, str]:
    """Check that Ollama is reachable and mistral is available."""
    try:
        import requests
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"Ollama returned HTTP {resp.status_code}"
        models = [m["name"] for m in resp.json().get("models", [])]
        mistral_models = [m for m in models if "mistral" in m.lower()]
        if not mistral_models:
            return False, (
                f"Ollama is running but 'mistral' is not yet pulled. "
                f"Available: {models}. Run: ollama pull mistral"
            )
        return True, f"OK (model: {mistral_models[0]})"
    except Exception as e:
        return False, f"Cannot reach Ollama at {OLLAMA_BASE}: {e}"


# ══════════════════════════════════════════════════════════════════════════
# Citation Parser
# ══════════════════════════════════════════════════════════════════════════

def parse_citations(
    answer_text: str,
    chunks: list[dict],
) -> tuple[str, list[dict]]:
    """
    Extract SOURCES: [...] line from the LLM answer.
    Returns (clean_answer, list_of_cited_chunk_dicts).
    """
    cited_chunks = []
    clean_answer = answer_text

    # Find "SOURCES: [1], [2], ..." line
    src_match = re.search(
        r"SOURCES\s*:\s*([\[\d\],\s]+)",
        answer_text,
        re.IGNORECASE,
    )
    if src_match:
        # Remove SOURCES line from the answer
        clean_answer = answer_text[: src_match.start()].rstrip()
        # Parse source numbers
        numbers = re.findall(r"\d+", src_match.group(1))
        for n in numbers:
            idx = int(n) - 1  # 1-based → 0-based
            if 0 <= idx < len(chunks):
                cited_chunks.append(chunks[idx])

    # Deduplicate cited chunks
    seen = set()
    deduped = []
    for c in cited_chunks:
        key = c.get("chunk_id", "")
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return clean_answer, deduped


# ══════════════════════════════════════════════════════════════════════════
# RAG Pipeline
# ══════════════════════════════════════════════════════════════════════════

class OlistRAGPipeline:
    """
    End-to-end RAG pipeline:
      query → embed → FAISS top-5 → build prompt → Mistral → parse citations
    """

    def __init__(self, top_k: int = 5, temperature: float = 0.1):
        self.top_k    = top_k
        self.retriever = FAISSRetriever()
        self.llm       = make_llm(temperature=temperature)
        self._chain    = self.llm | StrOutputParser()
        print(f"[OlistRAGPipeline] Ready | top_k={top_k} | model={OLLAMA_MODEL}")

    def retrieve(self, query: str) -> list[dict]:
        return self.retriever.search(query, top_k=self.top_k)

    def run(self, query: str) -> dict:
        """
        Full pipeline. Returns:
        {
            "query":          str,
            "query_type":     "rag",
            "answer":         str,
            "citations":      [{"chunk_id", "doc_name", "similarity", "text_preview"}],
            "retrieved_chunks": [...],
            "prompt":         str   (for debugging)
        }
        """
        chunks = self.retrieve(query)
        if not chunks:
            return {
                "query":      query,
                "query_type": "rag",
                "answer":     "No relevant context found in the knowledge base.",
                "citations":  [],
                "retrieved_chunks": [],
                "prompt":     "",
            }

        prompt_text = build_prompt(query, chunks)
        raw_answer  = self._chain.invoke(prompt_text)
        clean_ans, cited = parse_citations(raw_answer, chunks)

        citation_cards = [
            {
                "source_num":   chunks.index(c) + 1 if c in chunks else "?",
                "chunk_id":     c.get("chunk_id", ""),
                "doc_name":     c.get("doc_name", ""),
                "similarity":   round(c.get("similarity", 0.0), 4),
                "text_preview": c.get("text", "")[:200] + "...",
            }
            for c in cited
        ]

        return {
            "query":            query,
            "query_type":       "rag",
            "answer":           clean_ans.strip(),
            "citations":        citation_cards,
            "retrieved_chunks": chunks,
            "prompt":           prompt_text,
        }


# ── Quick self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok, msg = check_ollama()
    print(f"\nOllama status: {msg}")

    if ok:
        pipe = OlistRAGPipeline(top_k=5)
        result = pipe.run("Which region has the slowest average delivery time?")
        print("\n--- ANSWER ---")
        print(result["answer"])
        print("\n--- CITATIONS ---")
        for c in result["citations"]:
            print(f"  [{c['source_num']}] {c['chunk_id']} (doc: {c['doc_name']}, sim: {c['similarity']})")
    else:
        retriever = FAISSRetriever()
        chunks = retriever.search("Which region has the slowest delivery?", top_k=3)
        print("\n--- TOP-3 RETRIEVED CHUNKS (retrieval test, no LLM) ---")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] {c['chunk_id']} | sim={c['similarity']:.4f}")
            print(c["text"][:300])
