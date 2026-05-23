# 📊 Customer Lifetime Value (CLTV) Prediction Pipeline

An enterprise-grade, end-to-end Machine Learning pipeline designed to predict customer transaction frequencies and financial values over a rolling 90-day horizon. Utilizing transactional data from the UCI Online Retail dataset, this project implements structural probabilistic modeling to segment customers, evaluate marketing ROI, and mitigate customer churn.

---

## 🚀 Business Problem & Objectives
Traditional marketing focus metrics treat all customers equally based on historical transactions, leading to wasted marketing spend on low-value users. 

**This pipeline shifts the strategy from reactive to predictive by answering:**
1. How many purchases will a customer make in the next 90 days?
2. What is the expected average monetary value per transaction?
3. Which customer cohorts represent the highest long-term financial equity (VIPs vs. Churn risks)?

---

## 🛠️ Tech Stack & Frameworks
* **Language:** Python 3.11+
* **Probabilistic Modeling:** `lifetimes` (BG/NBD and Gamma-Gamma Fitter)
* **Data Engineering:** `pandas`, `numpy`
* **Visualization & Reporting:** `matplotlib`, `seaborn`, **Power BI**
* **Version Control & Automation:** Git, GitHub Architecture

---

## 📐 Project Architecture & Workflow

### 🔹 Phase 1: Data Cleansing & RFM Engineering
* Handled negative transaction quantities, unit pricing anomalies, and neutralized canceled orders (invoices starting with 'C').
* Engineered structural features aggregated at the individual customer level:
  * **Frequency:** Count of repeat purchases.
  * **Recency:** Age of the customer at their last purchase.
  * **T:** Customer age/tenure since their first purchase.
  * **Monetary Value:** Average spend per transaction.
* Generated output cache: `rfm_summary.csv`.

### 🔹 Phase 2: Statistical Modeling & Optimization
* **BG/NBD Model (Beta-Geometric/Negative Binomial Distribution):** Trained to model customer drop-out rates, assessing the mathematical probability that a customer is still active vs. inactive.
* **Gamma-Gamma Model:** Trained on repeat-purchase behavior to estimate the expected mean transaction value per customer segment.
* **CLTV Aggregator:** Combined the statistical distribution layers to compute individual expected values for the next 90 days.
* Generated output cache: `cltv_predictions.csv`.

### 🔹 Phase 3: Strategic Value Segmentation & ROI Matrix
Applied the 80/20 rule to divide the entire customer base into 4 data-driven cohorts for optimized business targeting:

| Segment | Distribution | Core Characteristics | Actionable Business Strategy |
| :--- | :--- | :--- | :--- |
| **VIP** | Top 10% | Premium spenders driving maximum revenue share. | Personalized white-glove service, loyalty perks. |
| **High Value** | Next 20% | Steady transaction volume with upselling headroom. | Tiered cross-selling, exclusive product drops. |
| **Medium Value**| Middle 30% | Moderate engagement; shows signs of fatigue. | Win-back email automations, behavior-driven discounts. |
| **Low Value** | Bottom 40% | Low transactional frequency; high churn probability. | Automated low-cost messaging; minimize marketing overhead. |

---

## 📈 File Structure
```text
├── rfm_summary.csv               # Processed RFM feature metrics (Day 1 Output)
├── cltv_predictions.csv          # Finalized 90-day predictions & CLTV scores (Day 2 Output)
├── ecommerce_cltv_pipeline.py    # Main training and execution pipeline script
├── model_validation.py           # Evaluation script checking goodness-of-fit
├── cltv_model_diagnostics.png    # Validation diagnostic graphs 
├── model_validation_summary.png  # Expected vs Actual calibration plotting
└── README.md                     # Documentation

