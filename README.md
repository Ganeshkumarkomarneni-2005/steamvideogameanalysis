# 🎮 Steam Game Intelligence — Portfolio Master Project

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/SQL-DuckDB-yellow.svg)](https://duckdb.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/)

An end-to-end analytics, machine learning, and AI portfolio project built on the Kaggle Steam dataset. This repository integrates data cleaning, exploratory visual analysis, reproducible DuckDB SQL queries, NLP sentiment classification, leakage-audited ML recommendation models, and a grounded AI analytics agent.

---

## 📌 Executive Summary & Key Findings

1. **Catalog & Genre Dominance**: Action (165 games, ~43.9M reviews) and RPG (81 games, ~14.5M reviews) dominate Steam catalog presence and overall player engagement.
2. **Commercial vs Review Rank Divergence**: Identified notable rank gaps (>30 rank divergence) between Sales Rank and Review Rank—highlighting commercial successes experiencing player dissatisfaction versus "hidden gem" titles with stellar player reception but lower sales velocity.
3. **ML Recommendation Prediction**: Achieved **0.9324 ROC-AUC** and **0.9424 F1-Score** using a TF-IDF + Logistic Regression pipeline predicting player recommendation status (`is_recommended`), outperforming the baseline (`0.8971 F1-Score`).
4. **Grounded AI Agent**: Implemented a read-only DuckDB SQL agent that converts natural language queries into safe, executed SQL code without data hallucination.

---

## 📁 Repository Structure

```
steam-game-intelligence/
├── data/
│   ├── README.md
│   └── processed/
│       ├── games_description_clean.csv
│       ├── games_ranking_clean.csv
│       └── steam_game_reviews_clean.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_and_business_analysis.ipynb
│   ├── 03_nlp_and_machine_learning.ipynb
│   └── 04_ai_analytics_agent.ipynb
├── sql/
│   └── business_queries.sql
├── models/
│   ├── README.md
│   └── recommendation_pipeline.joblib
├── dashboard/
│   └── steam_intelligence.pbix
├── images/
│   └── dashboard_screenshots/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Profile & Audit Trail

| Dataset | Raw Count | Clean Count | Unique Titles | Key Join Overlap | Audit Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `games_description.csv` | 290 rows | 290 rows | 290 | 100% (Base catalog) | 13 missing short descriptions flagged & retained. |
| `games_ranking.csv` | 672 rows | 672 rows | 303 | 97.62% match (656/672) | 13 unmatched titles identified as DLCs, content passes, or separate releases. |
| `steam_game_reviews.csv` | 992,153 rows | 992,153 rows | 242 | 93.26% match (925,244) | Processed in 100k chunks. 15 unmatched titles audited. |

---

## 🤖 Machine Learning Model Benchmarks

Target: Binary recommendation prediction (`is_recommended` = 1 vs 0).  
Data Split: 80/20 Train/Test split on 100,000 stratified review samples (Seed = 42).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Dummy Classifier)** | 81.35% | 81.35% | 100.00% | 0.8971 | 0.5000 |
| **TF-IDF + Logistic Regression** | **90.34%** | **91.55%** | **97.08%** | **0.9424** | **0.9324** |

### Top Sentiment Feature Weights
- **Positive Predictors**: `best` (+7.18), `great` (+5.59), `amazing` (+5.56), `10 10` (+4.58), `masterpiece` (+3.63).
- **Negative Predictors**: `boring` (-6.72), `worst` (-5.33), `refund` (-5.30), `worse` (-5.06), `unplayable` (-4.61), `greedy` (-3.87).

---

## 💡 Business Recommendations (Finding → Evidence → Action → Limitation)

1. **Recommendation 1: Monetization Friction Mitigation**
   - **Finding**: High Sales Rank titles occasionally experience low Review Rank due to player friction surrounding post-launch monetization.
   - **Evidence**: Top negative review keywords feature `refund` (-5.30), `greedy` (-3.87), and `pay` (-3.84).
   - **Action**: Publishers should re-evaluate battle pass structures and microtransaction pricing models before launch.
   - **Limitation**: Observational review text does not directly capture exact user spend amount per transaction.

2. **Recommendation 2: Target High-Engagement Sub-Genres**
   - **Finding**: Action-RPG and Tactical Strategy titles exhibit higher median playtime per player.
   - **Evidence**: Median playtime correlates positively with recommendation rate up to ~40 hours of gameplay.
   - **Action**: Developers should focus design investments on deep progression loops rather than superficial map size expansion.
   - **Limitation**: Playtime data reflects current user session logs and may be skewed by idle launcher hours.

---

## ⚙️ How to Reproduce (Google Colab & Local Python)

### Local Environment Setup
```bash
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL>
cd steam-game-intelligence

# 2. Initialize Virtual Environment & Install Dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Execute Pipelines
python scripts/01_data_cleaning.py
python scripts/02_eda_and_sql_analysis.py
python scripts/03_nlp_and_ml.py
python scripts/04_ai_analytics_agent.py
```

---

## 📜 Dataset Usage & Licensing
This project utilizes the publicly available Kaggle Steam dataset. Raw CSV files are excluded from git commits via `.gitignore` in accordance with repository size best practices and dataset usage guidelines.
