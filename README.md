# 🛒 E-Commerce Intelligence Platform

**End-to-end customer & delivery analytics on 100K+ real e-commerce orders — RFM segmentation, delivery delay prediction, and SHAP explainability.**

🔗 **Live App:** [ecommerce-intelligence-ai.streamlit.app](https://ecommerce-intelligence-ai.streamlit.app)

---

## Overview

This project turns raw transactional data from a real Brazilian e-commerce marketplace into actionable business intelligence. It combines SQL-based data engineering, machine learning, unsupervised customer segmentation, and model explainability into a single deployed application.

The goal: simulate what a Data Scientist / ML Engineer would actually build at an e-commerce company — not just a notebook with a model, but a full pipeline from raw data to a stakeholder-facing tool.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Warehouse | Google BigQuery |
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | XGBoost, scikit-learn |
| Explainability | SHAP |
| Clustering | KMeans, StandardScaler |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — ~100K orders across 8 relational tables (orders, customers, products, sellers, payments, reviews, geolocation).

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   BigQuery SQL   │ --> │  Python / ML      │ --> │  SHAP            │ --> │  Streamlit App    │
│  RFM, features,   │     │  XGBoost, KMeans  │     │  Explainability   │     │  Interactive UI    │
│  time-safe labels │     │                    │     │                   │     │                    │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
```

---

## Layer 1 — Data Engineering (BigQuery SQL)

- Built an **RFM (Recency, Frequency, Monetary)** feature table using window functions (`NTILE`) and date arithmetic
- Engineered a **time-safe delivery delay dataset** — features computed only from orders before a cutoff date to avoid data leakage
- Joined across 5+ relational tables (orders, customers, payments, products, sellers, reviews) to build enriched feature sets

**Key technique:** time-based train/test separation at the SQL level — features are computed from a past window, labels from a future window, to ensure the model never sees information it wouldn't have at prediction time.

---

## Layer 2 — Machine Learning

### Delivery Delay Prediction (XGBoost Regression)

Predicts how many days early/late an order will arrive based on order, product, and shipping characteristics.

| Metric | Score |
|---|---|
| MAE | 4.28 days |
| R² | 0.49 |
| RMSE (final) | 6.00 |

**Top predictive features (via SHAP):**
1. Estimated delivery window
2. Order month (seasonality)
3. Customer state (geographic distance from sellers)
4. Freight value
5. Seller state

### Customer Segmentation (KMeans)

Segmented ~96K customers into 4 behavioral clusters using RFM features, scaled with `StandardScaler` and validated via the elbow method.

| Segment | Description |
|---|---|
| **Champions** | Recent, high-spending customers |
| **High Value** | High spenders regardless of recency |
| **Casual** | Low spenders, moderate recency |
| **Lost** | Inactive for 300+ days, low spend |

---

## Layer 3 — Explainability (SHAP)

Every prediction from the delivery model is explainable at two levels:

- **Global** — which features matter most across all predictions (summary plot)
- **Local** — why a specific order was predicted to be early/late (force plot)

This moves the project beyond a black-box model into something a business stakeholder can trust and act on.

---

## Layer 4 — Streamlit Application

A 3-page interactive dashboard:

1. **Customer Lookup** — search any customer ID, view their RFM profile, segment, and a recommended retention action
2. **Segment Overview** — distribution and average spend across all 4 customer segments
3. **Delivery Predictor** — input order/product details and get a live delay prediction with SHAP explanation

---

## Project Structure

```
ecommerce-intelligence/
├── app.py                       # Streamlit application
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version for deployment
├── data/
│   └── rfm_segments.csv         # Customer RFM + segment labels
├── models/
│   ├── delivery_model.pkl       # XGBoost regressor
│   ├── delivery_explainer.pkl   # SHAP TreeExplainer
│   ├── kmeans_model.pkl         # KMeans clustering model
│   └── kmeans_scaler.pkl        # StandardScaler for RFM features
├── sql/                         # BigQuery feature engineering queries
├── notebooks/                   # EDA and model training notebooks
└── README.md
```

---

## Key Engineering Decisions

**Avoiding data leakage in churn modeling** — An initial churn model scored a suspicious 1.0 AUC. Root cause: the churn label was derived directly from the same recency feature used to predict it. Fixed by separating feature and label windows in time, and ultimately pivoting to delivery delay prediction, which had a cleaner causal structure and no repeat-customer sparsity problem (97% of Olist customers are one-time buyers).

**Handling class imbalance** — Used `scale_pos_weight` in XGBoost to account for skewed class distributions rather than naively oversampling.

**Feature scaling for KMeans** — Standardized RFM features before clustering since KMeans is distance-based; without this, `recency_days` (range 0–700) would have dominated `monetary` (log-scale, range 0–10).

---

## Results Summary

| Model | Type | Metric | Score |
|---|---|---|---|
| Delivery Delay Predictor | XGBoost Regression | MAE | 4.28 days |
| Delivery Delay Predictor | XGBoost Regression | R² | 0.49 |
| Customer Segmentation | KMeans (K=4) | Validated via | Elbow method |

---

## Future Improvements

- Add a review-score / satisfaction prediction model
- Incorporate geolocation data for distance-based delivery features
- A/B test framework for retention campaign effectiveness
- Real-time data refresh via scheduled BigQuery queries

---

## Author

Arpit — CS student, Manipal Institute of Technology
Built as a portfolio project for Data Scientist / ML Engineer roles.
