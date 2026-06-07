# 🤖 Advanced Data Science & Machine Learning Portfolio
### *Enterprise Predictive Pipelines, Conversational BI Architectures & Time Series Analytics*

Welcome to my production-ready engineering repository. This workspace hosts modular, end-to-end analytical solutions designed to tackle complex corporate data bottlenecks—ranging from customer unit economics and semantic search to macroeconomic forecasting.

---

## 📁 Executive Dashboard & Project Index

| Project Pipeline | Core Technology Stack | Primary Business Target | Key Evaluation Metric |
| :--- | :--- | :--- | :--- |
| **[1. Customer Lifetime Value Prediction](./CLTV_PROJ)** | Python, `lifetimes`, Probabilistic Modelling, Power BI | Monetisation & Churn Mitigation | Expected 90-Day Gross Margin |
| **[2. Conversational RAG Analytics Agent](./RAG_AGENT)** | LangChain, FAISS Vector DB, Mistral-7B via Ollama | Zero-Friction Conversational BI | >90% Faithfulness (RAGAS) |
| **[3. Predictive Sales Time Series Analysis](./Time%20Series%20Analysis)** | Power BI, Star Schema Modeling, Trend Filters | Procurement & Supply Risk Mitigation | Mean Absolute Error (MAE) KPI |
| **[4. Credit Card Fraud Detection Pipeline](./Fraud%20Detection)** | Python, XGBoost, SHAP, SMOTE, LightGBM | Financial Crime Prevention & Operational Cost Reduction | ROC-AUC 0.977 · $26.6M Annual Savings |

---

## 📈 Deep-Dive Project Breakdown

### 📊 1. Customer Lifetime Value (CLTV) Prediction Pipeline
* [cite_start]**The Business Problem:** Marketing teams treat all customer acquisition retention spend uniformly [cite: 103][cite_start], leading to high acquisition burn rates on low-value users[cite: 141].
* [cite_start]**Technical Execution:** Implemented a **BG/NBD model** to mathematically isolate active vs. churned users [cite: 118, 119] [cite_start]paired with a statistical **Gamma-Gamma Fitter** to simulate future transaction valuations over a 90-day horizon[cite: 122, 123, 128]. [cite_start]Developed a native **Power BI dashboard** to parse the predictive models[cite: 142].
* **💡 SO WHAT? (Business Impact)[cite_start]:** **Segments the entire customer registry into 4 actionable priority tiers (VIP, High, Medium, Low)**[cite: 139]. [cite_start]This permits management to automate high-conversion automated loyalty campaigns for the top 10% premium bracket [cite: 138, 146] [cite_start]while preventing marketing overspend on high-churn customer segments[cite: 141, 146].

---

### 🧠 2. Conversational Business Intelligence Agent (RAG Architecture)
* **The Business Problem:** Non-technical corporate leaders suffer heavily from data access friction—relying on manual analyst SQL turnaround cycles to fetch simple operational metrics[cite: 161].
* [cite_start]**Technical Execution:** Constructed a local **Retrieval-Augmented Generation (RAG) agent** over high-volume transactional summaries[cite: 159, 161]. [cite_start]Leveraged `all-MiniLM-L6-v2` for semantic indexing into a persistent **FAISS Vector DB**[cite: 166]. [cite_start]Integrated an intelligent **deterministic query router** to direct basic queries via direct Pandas vector calculations, passing complex analytical queries to a locally hosted **Mistral-7B instance**[cite: 169].
* **💡 SO WHAT? (Business Impact)[cite_start]:** **Slashes time-to-insight from hours to seconds** by allowing executives to query operational realities using plain natural language[cite: 161]. [cite_start]Cost-optimized routing architecture cuts token compute costs while **guaranteeing 100% auditable source-chunk citations** to completely eliminate artificial intelligence hallucinations[cite: 162, 163, 169]. [cite_start]Evaluated rigorously via **RAGAS framework** to benchmark system robustness[cite: 171].

---

### 📉 3. Trend-Focused Sales Forecasting & Time Series Analytics
* [cite_start]**The Business Problem:** Raw daily transactional data contains high seasonal fluctuations ("noise") [cite: 179, 182][cite_start], leading to procurement errors and inaccurate inventory planning[cite: 193].
* [cite_start]**Technical Execution:** Architected a high-contrast corporate dashboard leveraging an optimized **Star Schema relational model**[cite: 181, 195]. [cite_start]Implemented a continuous **30-Day Moving Average trend line** to strip daily noise metrics and deployed a **90-day forward predictive horizon** flanked by **95% Confidence Shaded Error Bands**[cite: 182, 183, 191, 193].
* **💡 SO WHAT? (Business Impact)[cite_start]:** **Transitions corporate strategy from reactive monitoring to proactive supply-chain planning.** Explicitly flags extreme statistical variances (e.g., Q4 outliers) for subsequent algorithmic retraining loops[cite: 184, 189]. [cite_start]Deployed with contextual post-actual filtration logic so procurement officers can directly project risk parameters for **"Best Case" vs "Worst Case" inventory stock ordering scenarios**[cite: 186, 187, 193].

---

### 🔍 4. Credit Card Fraud Detection Pipeline
* **The Business Problem:** Financial institutions absorb massive chargeback losses from undetected card fraud while simultaneously alienating legitimate customers through excessive false-alarm friction—a dual cost vector that no threshold-agnostic classifier can resolve.
* **Technical Execution:** Engineered a full preprocessing pipeline over **283,726 transactions** addressing severe class imbalance (0.17% fraud) via **SMOTE oversampling**—applied exclusively to training data to prevent leakage. Benchmarked **4 models (XGBoost, Random Forest, LightGBM, Logistic Regression)** with stratified splits. Executed a **cost-weighted threshold optimisation** ($50/FP · $389/FN) to locate the operationally optimal decision boundary at `0.86`. Deployed **SHAP explainability** to surface the top 5 fraud-signal PCA components (V14, V4, V12, V10, V11) driving model decisions.
* **💡 SO WHAT? (Business Impact):** **XGBoost achieved ROC-AUC 0.977 with 99.88% accuracy**—threshold-tuned to deliver **80% fraud recall at a 0.014% false-alarm rate**, keeping analyst review queues operationally viable. Business impact modelling across **51.8 million annual transactions** projects a **$26,601,924 annual saving (78.9% cost reduction)** versus a no-model baseline of $33.7M/year. Precision at optimal threshold reaches **90.5%**—9 in 10 flagged transactions are genuine fraud.

---
*Developed and Maintained by [Bhawna Kaushik](https://github.com/bhawna407) — Data Analyst & Engineer
