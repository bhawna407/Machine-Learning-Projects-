# 🧠 Conversational Analytics Agent with RAG Architecture

An enterprise-grade Retrieval-Augmented Generation (RAG) analytics assistant engineered to allow non-technical business users to query high-volume sales and operations metrics using natural language. This system processes transactional text logs, builds a localized vector space, and utilizes an LLM to serve answers backed by verifiable source citations.

---

## 🚀 Business Core Objective
Traditional business intelligence requires SQL knowledge or manual dashboard deep-dives. This system shifts the analytics workflow to conversational interfaces, fetching multi-dimensional context instantly from raw operations data without any AI hallucinations.

---

## 🛠️ Deep Tech Stack & Frameworks
* **Orchestration & Pipeline:** LangChain
* **Vector Store & Indexing:** FAISS (Facebook AI Similarity Search)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Local LLM Engine:** Mistral-7B via Ollama
* **Data Layer:** Pandas, NumPy
* **User Interface:** Streamlit UI Components
* **Evaluation Framework:** RAGAS (Faithfulness, Context Recall, Answer Relevancy)

---

## 📐 3-Day Implementation Architecture

### 📅 Day 1: Data Synthesis, Semantic Chunking & Vector DB Setup
* **Data Transformation:** Processed the raw Brazilian E-Commerce Olist Dataset from Kaggle, aggregating unstructured information into highly structured text summaries (Monthly Sales, Seller Performance, Regional Delivery Latency, Review Sentiment Themes).
* **Chunking Strategy:** Structured documents into dense `200-400 token chunks` with custom overlap parameters to maintain context at boundaries.
* **Vectorization:** Encoded chunks through `all-MiniLM-L6-v2` and built a disk-persistent FAISS index structure.
* **Evaluation Framework:** Synthesized a diagnostic 20-question Q&A evaluation matrix covering complex reasoning types (Simple stats, Trend analysis, Comparison, Causal reasoning).

### 📅 Day 2: LLM Integration, Query Routing & Interface Draft
* **Local Deployment:** Configured Mistral-7B locally via Ollama to guarantee absolute data privacy and offline operational capabilities.
* **Smart Hybrid Query Routing:** Engineered an intelligent logical layer that bypasses the expensive RAG pipeline for simple arithmetic queries, executing direct Python data calculations instead, while routing analytical requests to the vector store.
* **Citation Mechanism:** Built a custom extraction layer that maps generated text answers back to the retrieved underlying source documents, ensuring transparency.

### 📅 Day 3: RAGAS Evaluation & Streamlit Polishing
* **Framework Benchmarking:** Executed full automated evaluations assessing system *Faithfulness* (grounded answers), *Context Recall* (retrieval capability), and *Answer Relevancy*.
* **UI Delivery:** Wrapped the localized pipeline into a clean Streamlit interface with sidebar metadata controls, tracking failure cases to continually enhance response boundaries.

---

## 📁 Folder Structure & Pipeline Artifacts
```text
├── DAY_1/
│   └── output/
│       ├── faiss_index.bin       # Serialized Vector Database Index
│       ├── chunk_store.json      # Mapping store for semantic text blocks
│       └── eval_set_20q.json     # 20-Question Q&A Evaluation Spreadsheet
├── app.py                        # Streamlit UI Dashboard Interface Code
├── rag_pipeline.py               # Core LangChain & Ollama orchestration logic
├── query_router.py               # Semantic query sorting framework
└── README.md                     # Documentation
