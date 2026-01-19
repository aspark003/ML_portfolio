Risk Intelligence Platform — Unsupervised Anomaly Detection & Severity Scoring Engine

Overview

This project is a full end‑to‑end risk intelligence and anomaly detection platform built in Python using modern machine‑learning techniques. 
It is designed to ingest raw, messy business data, clean and engineer features, run multiple unsupervised anomaly detection models, and produce a single unified severity score for every record.

The system is model‑agnostic and can be applied to financial, operational, cybersecurity, compliance, or fraud‑style datasets.

What This Platform Does

1. Ingests raw tabular data (numeric + categorical)
2. Cleans and preprocesses using a production‑grade pipeline
3. Applies dimensionality reduction (PCA)
4. Runs multiple anomaly detection models
5. Normalizes and fuses all model outputs
6. Generates a final severity score and risk levels
7. Exports results for BI dashboards and analytics

Models Used

This platform uses an ensemble of industry‑grade unsupervised anomaly detection models:

 Model                                                                  

DBSCAN -- Density‑based clustering & noise detection          
OPTICS -- Reachability‑based density modeling                 
HDBSCAN -- Stability‑based clustering with probability scoring 
HDBSCAN Outlier Score -- Outlier ranking within clusters                     
Local Outlier Factor (LOF) -- Local density deviation                             
Isolation Forest -- Tree‑based isolation of anomalies                   
One‑Class SVM -- Boundary‑based anomaly detection                    

Each model produces its own anomaly signal which is normalized and converted into a severity score

Severity Scoring Engine

All models are fused into a single risk engine:

Raw Model Scores
      
Min‑Max Normalization
      
Severity Inversion (when needed)
      
Model Fusion (Density + Isolation + SVM)
      
Final Severity Score
      
Quantile Risk Levels


Risk Levels

High -- Top 25% 
Medium -- Middle 50%                     |
Low -- Bottom 25%                     |

This ensures a stable and interpretable ranking system suitable for risk intelligence investigation workflows.

Feature Engineering Pipeline

A production‑ready preprocessing pipeline is used:

Numeric imputation (median)
Categorical imputation (most frequent)
Min‑Max scaling
One‑hot encoding
ColumnTransformer

This ensures the platform can handle:

Missing values
Mixed data types
Dirty real‑world datasets

Output

The final output is a fully enriched dataset containing:

Model labels
Confidence scores
Severity scores
Risk levels
Final risk severity score
Final risk severity levels

This output is designed for direct ingestion into BI tools such as Power BI.

Example Dashboard

The system integrates directly with a Power BI risk intelligence dashboard featuring:
Risk level button
Severity distribution donut charts
Total exposure gauges

This enables analysts to move from macro risk posture → individual anomalous records in seconds.

Architecture
Raw Data
Preprocessing Pipeline
PCA
Anomaly Model Ensemble
Severity Fusion Engine
Risk Level
Analytics & Dashboarding

Technologies
Python 3.11+
pandas, NumPy
scikit‑learn
HDBSCAN
Power BI

Use Cases

Fraud detection
Insider threat
Financial risk analytics
Compliance monitoring
Cybersecurity anomaly detection
Operational risk

Why This Matters

Most anomaly systems rely on a single model. This platform uses model consensus, dramatically reducing false positives while improving detection of real risk.

This architecture mirrors how enterprise fraud, SOC, and compliance platforms operate in production environments.

Author

Antonio Park
Machine Learning Engineer — Risk & Anomaly Detection

GitHub: [https://github.com/aspark003/ML_portfolio](https://github.com/aspark003/ML_portfolio)
