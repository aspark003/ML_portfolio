# Risk Intelligence Platform  
## Unsupervised Anomaly Detection & Severity Scoring Engine

## Overview

This project is a **full end-to-end risk intelligence platform** built in Python using modern machine learning techniques.  
It ingests raw, messy tabular data, performs robust preprocessing and feature engineering, applies multiple unsupervised anomaly detection models, and produces a **unified severity score and risk level** for each record.

The platform is **model-agnostic** and applicable to financial, operational, cybersecurity, compliance, and fraud-oriented datasets.

---

## What This Platform Does

- Ingests raw tabular data (numeric + categorical)
- Cleans and preprocesses data using a production-grade pipeline
- Applies dimensionality reduction (PCA)
- Executes an ensemble of unsupervised anomaly detection models
- Normalizes and fuses model outputs into a single severity signal
- Assigns quantile-based risk levels
- Exports results for interactive BI dashboards

---

## Models Used

This platform implements an ensemble of industry-standard unsupervised detection techniques:

| Model | Purpose |
|------|--------|
| DBSCAN | Density-based clustering and noise detection |
| OPTICS | Reachability-based density modeling |
| HDBSCAN | Stability-based clustering with probability scoring |
| HDBSCAN Outlier Score | Outlier ranking within clusters |
| Local Outlier Factor (LOF) | Local density deviation |
| Isolation Forest | Tree-based isolation of anomalies |
| One-Class SVM | Boundary-based anomaly detection |

Each model generates an independent anomaly signal that is normalized and incorporated into the severity engine.

---

## Severity Scoring Engine

All model outputs are fused into a single risk scoring pipeline:

1. Raw model scores  
2. Min–Max normalization  
3. Score inversion (where applicable)  
4. Model fusion (density + isolation + boundary signals)  
5. Final severity score  
6. Quantile-based risk level assignment  

This produces a **stable, interpretable ranking** suitable for investigation and prioritization workflows.

---

## Risk Levels

| Risk Level | Quantile |
|----------|---------|
| Critical | Top 5% (≥ 0.95) |
| High | Top 25% (≥ 0.85) |
| Medium | Middle 50% (≥ 0.60) |
| Low | Bottom 25% |

This structure ensures consistent prioritization across datasets and refresh cycles.

---

## Feature Engineering Pipeline

The preprocessing pipeline is designed for **real-world data robustness** and includes:

- Numeric imputation (median)
- Categorical imputation (most frequent)
- Min–Max scaling
- One-hot encoding using `ColumnTransformer`

The pipeline is resilient to:
- Missing values
- Mixed data types
- Noisy and incomplete datasets

---

## Power BI Dashboard Integration

The platform exports analytics-ready outputs for Power BI, enabling interactive risk exploration.

Dashboard features include:
- Master risk-level slicer controlling all visuals
- KPI cards for:
  - Full automation detection
  - Anomaly detection
  - Risk analytics detection
- Trend axes per severity signal
- Drill-down from aggregate risk posture to individual records

This allows analysts to move from **macro risk visibility to individual anomalies in seconds**.

---

## Architecture

Raw Data
→ Preprocessing Pipeline
→ PCA
→ Anomaly Model Ensemble
→ Severity Fusion Engine
→ Risk Level Assignment
→ Analytics & Dashboarding


---

## Technologies

- Python 3.11+
- pandas, NumPy
- scikit-learn
- HDBSCAN
- Power BI

---

## Use Cases

- Fraud detection
- Insider threat detection
- Financial risk analytics
- Compliance monitoring
- Cybersecurity anomaly detection
- Operational risk analysis

---

## Why This Matters

Most anomaly detection systems rely on a **single model**, leading to brittle results and high false positives.  
This platform uses **model consensus**, improving robustness, stability, and real-world applicability.

The architecture mirrors how **enterprise fraud, SOC, and compliance platforms** operate in production environments.

---

## Author

**Antonio Park**  
Machine Learning Engineer — Risk & Anomaly Detection  

GitHub: https://github.com/aspark003/ML_portfolio
