https://app.fabric.microsoft.com/groups/me/dashboards/1c7a9c3e-5f14-4414-90ed-923cdc4e7027?experience=fabric-developer

Anomaly Detection & Fraud Risk Screening Dashboard

Unsupervised ML + Regression Deviation Analysis with Power BI

Project Overview

This project implements an end-to-end anomaly detection and risk screening system for financial and ledger-style data.

The system is designed to surface unusual patterns and material deviations, assign severity, and prioritize records for audit and investigative review using an interactive Power BI dashboard.

The goal is not fraud prediction, but risk discovery, validation, and prioritization.

Modeling Philosophy

The system separates detection from decision-making:

Machine learning identifies what is abnormal

Severity quantifies how significant the deviation is

Binary review flags drive human investigation

This design mirrors real-world audit and compliance workflows.

Modeling Components

Unsupervised Anomaly Detection (Pattern-Based)

To identify structural anomalies, the system uses model consensus across multiple density-based algorithms:

Algorithms implemented:

DBSCAN – density-based outlier detection

OPTICS – variable-density clustering

HDBSCAN – hierarchical density-based clustering

PCA – dimensionality reduction for clustering stability

(Isolation Forest included as a reference model)

Each model produces:

cluster labels

anomaly identifiers

Model outputs are aggregated, not used independently.

Consensus-Based Anomaly Severity

Model agreement is combined into a single score:

Anomaly Agreement Strength

DBSCAN + OPTICS + HDBSCAN


Severity Levels

0 → None

1 → Low

2 → Medium

3 → High

This answers:

How confident are we that this record is structurally unusual?

Regression-Based Deviation Analysis (Value-Based)

Regression is used only to quantify deviation magnitude, not classification.

For each record:

Actual value is compared to a model-expected value

Residuals (actual − predicted) measure financial deviation

Residuals are:

bucketed into severity levels using quantiles

collapsed into a binary investigation flag

This answers:

How far off is this value from what would be expected?

Investigation & Fraud Screening Logic

Both anomaly severity and regression residual severity feed into review gates:

Investigation Required → needs analyst review

Fraud Candidate → potential fraud (screening only)

No record is labeled as confirmed fraud.
The system identifies candidates for review, not intent.

Pipeline Architecture
Data Ingestion

Supports Excel, CSV, TSV, JSON, TXT

Automatic validation and safe loading

Original data preserved for audit traceability

Preprocessing

Median imputation (numeric)

Constant imputation (categorical)

Min-Max scaling

One-Hot Encoding

Managed via ColumnTransformer

Modeling

Independent pipelines for DBSCAN, OPTICS, HDBSCAN

PCA applied per model (configurable components)

Cluster evaluation using:

Silhouette Score

Calinski-Harabasz Index

Davies-Bouldin Index

Dashboard Features (Power BI)

Interactive anomaly & deviation table

Row-level highlighting for:

anomaly severity

regression deviation severity

investigation status

KPI cards for:

None / Low / Medium / High anomaly counts

Records requiring investigation

Drill-down by:

fund

organization

project

account code

Designed to support audit review, compliance checks, and prioritization.

Output Artifacts

original.csv – enriched dataset with all anomaly and regression flags

Model evaluation metrics logged per algorithm

Power BI dashboard connected to finalized dataset

Example Model Metrics
Model	Silhouette	Calinski-Harabasz	Davies-Bouldin
DBSCAN	~0.70	~47	~1.00
OPTICS	~0.50	~15	~1.56
HDBSCAN	~0.72	~90	~0.68

HDBSCAN showed the strongest separation, reinforcing the multi-model consensus approach.

Key Takeaways

Unsupervised models identify risk signals, not conclusions

Model consensus reduces false positives

Regression residuals quantify material impact

Dashboards convert ML outputs into actionable review workflows

Use Cases

Fraud screening (candidate identification)

Financial audit support

Budget and expenditure anomaly discovery

Compliance monitoring

Risk prioritization

Tech Stack

Python (pandas, numpy, scikit-learn)

PCA, DBSCAN, OPTICS, HDBSCAN

Power BI / Fabric

GitHub for version control
