Risk Intelligence Platform — Unsupervised Anomaly Detection & Severity Scoring Engine

Overview

This project is a full end‑to‑end risk intelligence and anomaly detection platform built in Python using modern machine‑learning techniques.
It ingests raw, messy business data, cleans and engineers features, runs multiple unsupervised anomaly detection models, and produces a single unified severity score and risk level for every record.

The system is model‑agnostic and can be applied to financial, operational, cybersecurity, compliance, or fraud-style datasets.

What This Platform Does

Ingests raw tabular data (numeric + categorical)

Cleans and preprocesses using a production‑grade pipeline

Applies dimensionality reduction (PCA)

Runs multiple anomaly detection models

Normalizes and fuses all model outputs

Generates a final severity score and risk levels

Exports results for interactive BI dashboards

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

All model outputs are fused into a single risk engine:

Raw Model Scores

Min-Max Normalization

Severity Inversion (when needed)

Model Fusion (Density + Isolation + SVM)

Final Severity Score

Quantile-based Risk Levels


Risk Levels

Critical -- Top 5% / Quantile 0.95
High -- Top 25%  / Quantile 0.85
Medium -- Middle 50% / Quantile 0.60                    |
Low -- Bottom 25%                     |

This ensures a stable and interpretable ranking system suitable for risk intelligence investigation workflows.

Feature Engineering Pipeline

Production-ready preprocessing pipeline handles:

Numeric imputation (median)

Categorical imputation (most frequent)

Min-Max scaling

One-hot encoding (ColumnTransformer)

It ensures robustness to:

Missing values

Mixed data types

Dirty real-world datasets

Power BI Dashboard Integration

The platform integrates with an interactive Power BI dashboard, featuring:

Risk level identifier slicer → → automatically update per selection

Full Automation Detection = KPI card

Anomaly Detection = KPI card

Risk Analytics Detection = KPI card

This allows analysts to move from macro risk posture → individual anomalous records in seconds.

KPI Card Setup:

Value: Count of ID

Trend axis: column corresponding to each KPI (risk detection level, density severity level, decision severity level)

Target: Count of ID

Slicer: one master slicer for risk levels, controlling all KPI cards

Architecture

Raw Data → Preprocessing Pipeline → PCA → Anomaly Model Ensemble → Severity Fusion Engine → Risk Level → Analytics & Dashboarding


GitHub: [https://github.com/aspark003/ML_portfolio](https://github.com/aspark003/ML_portfolio)

Technologies

Python 3.11+

pandas, NumPy

scikit-learn

HDBSCAN

Power BI

Use Cases

Fraud detection

Insider threat detection

Financial risk analytics

Compliance monitoring

Cybersecurity anomaly detection

Operational risk

Why This Matters

Most anomaly systems rely on a single model. This platform uses model consensus, reducing false positives and improving detection of real risk.

The architecture mirrors how enterprise fraud, SOC, and compliance platforms operate in production.

Author

Antonio Park
Machine Learning Engineer — Risk & Anomaly Detection

GitHub: https://github.com/aspark003/ML_portfolio
