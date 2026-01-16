Anomaly Intelligence Dashboard
ML-Powered Anomaly Detection and Risk Scoring Platform
Overview

The Operation Risk Intelligence Dashboard is an end-to-end anomaly detection and operational risk monitoring system built using unsupervised machine learning and Power BI.

The platform detects abnormal behavior in transactional data, assigns multi-layer risk scores, and produces an interactive intelligence dashboard for monitoring, investigation, and decision support.

This system is designed to reflect real-world fraud, operational risk, and anomaly surveillance platforms used in enterprise environments.

Objectives

Detect anomalous transactions using unsupervised machine learning

Generate normalized business-friendly risk scores

Classify operational severity using ensemble logic

Provide an interactive intelligence dashboard

Support investigation and monitoring workflows

Machine Learning Architecture

The platform uses a three-layer detection stack:

Layer	Model	Purpose
Global Risk Engine	Isolation Forest	Global anomaly detection
Local Anomaly Engine	Local Outlier Factor	Neighborhood density detection
Structural Engine	PCA Reconstruction Error	Structural anomaly detection
Business Aggregation	Ensemble Logic	Final severity classification
Detection and Scoring Pipeline
Global Risk Engine (Isolation Forest)

Detects globally rare behavior patterns.

Outputs:

Isolation labels

Decision scores

Risk score (normalized)

Risk level (Low, High, Critical)

Local Risk Engine (Local Outlier Factor)

Detects neighborhood-based anomalies.

Outputs:

Local outlier labels

Local outlier scores

Local risk level (Low, High, Critical)

Structural Anomaly Engine (PCA)

Detects violations of normal data structure.

Outputs:

PCA reconstruction error

Scaled PCA score

PCA level (Low, High, Critical)

Ensemble Severity Engine

Final operational decision based on agreement across all three detection layers.

Logic:

Critical: All three engines agree

High: At least two engines agree

Low: Otherwise

Outputs:

Severity level (Low, High, Critical)

Power BI Intelligence Dashboard

Dashboard Name:
Operation Risk Intelligence Dashboard

Capabilities:

Interactive risk controllers using donut charts

Severity-level filtering

Total population KPI

Multi-layer anomaly filtering

Investigation-ready workflow

Each donut chart acts as a controller and filters the entire page. Filters stack together to support multi-layer investigation logic.

Output Dataset

The system produces a fully enriched dataset containing:

Column	Description
id	Unique record ID
Isolation risk scores	Normalized global risk score
Isolation risk level	Global risk classification
Local Outlier Scores	Density anomaly score
Local Risk Level	Local anomaly classification
PCA Error	Structural reconstruction error
PCA Level	Structural anomaly classification
Severity Level	Final operational decision
Technologies Used

Python

Pandas

NumPy

Scikit-learn

Isolation Forest

Local Outlier Factor

PCA

Power BI Desktop

How to Run

Run the detector script:

python detector.py


The script will:

Load the dataset

Train anomaly detection models

Generate multi-layer risk scores

Output a fully enriched dataset for Power BI

System Architecture

Raw Data
→ Isolation Forest → Risk Score → Risk Level
→ Local Outlier Factor → Local Risk Level
→ PCA Error → PCA Level
→ Ensemble Logic → Severity Level
→ Power BI Intelligence Dashboard

Use Cases

Fraud detection

Financial anomaly detection

Insider threat detection

Operational monitoring

Risk surveillance

Compliance auditing

Author

Antonio Park
Machine Learning Engineer
Anomaly Detection and Risk Analytics
