https://app.fabric.microsoft.com/groups/me/dashboards/1c7a9c3e-5f14-4414-90ed-923cdc4e7027?experience=fabric-developer




Anomaly Detection & Fraud Risk Dashboard

Unsupervised ML Pipeline + Power BI Analytics

Project Overview

This project implements an end-to-end unsupervised anomaly detection system for financial and ledger-style data.
The pipeline detects unusual spending patterns using multiple density-based clustering algorithms, aggregates anomaly signals, and presents results in an interactive Power BI dashboard designed for audit, compliance, and financial oversight use cases.

The goal is not prediction, but risk discovery, validation, and prioritization.

Modeling Approach

The system uses model consensus, not a single algorithm, to reduce false positives and improve confidence.

Algorithms implemented:

DBSCAN – density-based anomaly detection

OPTICS – variable density clustering for irregular structures

HDBSCAN – hierarchical density-based clustering

PCA – dimensionality reduction for clustering stability

Isolation Forest (optional reference model)

Each model produces:

cluster labels

anomaly identifiers

categorical flags

These are combined into a unified severity score.

⚙️ Pipeline Architecture

1. Data Ingestion

Supports Excel, CSV, TSV, JSON, and TXT

Automatic validation and safe loading

Preserves original data for audit traceability

2. Preprocessing

Median imputation for numeric features

Constant imputation for categorical features

Min-Max scaling

One-Hot Encoding

Managed through ColumnTransformer

3. Modeling

PCA applied per model (configurable components)

Independent pipelines for:

DBSCAN

OPTICS

HDBSCAN

Cluster evaluation using:

Silhouette Score

Calinski-Harabasz Index

Davies-Bouldin Index

4. Consensus Scoring

Total Identifiers = DBSCAN + OPTICS + HDBSCAN
Severity Levels:
0 → None
1 → Low
2 → Medium
3 → High

📈 Dashboard Features (Power BI)

Interactive anomaly table

Row-level anomaly highlighting

Severity cards (None / Low / Medium / High)

Drill-down by:

fund

organization

project

account code

Supports audit review and investigation workflows

Output Artifacts

original.csv – enriched dataset with all anomaly labels

Model metrics printed per algorithm

Power BI report connected to final dataset

🧪 Model Performance (Example)
Model	Silhouette	Calinski-Harabasz	Davies-Bouldin
DBSCAN	~0.70	~47	~1.00
OPTICS	~0.50	~15	~1.56
HDBSCAN	~0.72	~90	~0.68

HDBSCAN provided the strongest clustering separation, reinforcing the multi-model consensus strategy.

Key Takeaways

Unsupervised models must be interpreted, not trusted blindly

Consensus across models dramatically improves reliability

PCA + density methods are effective for financial anomaly detection

Dashboards turn models into decision tools, not experiments

Use Cases

Fraud detection

Financial audit support

Budget anomaly discovery

Compliance monitoring

Risk prioritization

Tech Stack

Python (pandas, scikit-learn, numpy)

PCA, DBSCAN, OPTICS, HDBSCAN

Power BI

GitHub version control
