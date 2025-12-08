Fraud Detection & Anomaly Detection System

An end-to-end machine learning system using PCA, DBSCAN, OPTICS, and HDBSCAN to detect anomalies, assign fraud risk categories, and generate explainable outputs.
Integrated with an enterprise-style Power BI dashboard for real-time monitoring, KPIs, and automated alerts.

Project Overview

This project builds a full anomaly detection pipeline designed to identify fraudulent or unusual vendor transactions.
Using a combination of dimensionality reduction, density-based clustering, and risk scoring, this system creates a multi-model fraud assessment that mimics real-world enterprise fraud detection engines.

Key Machine Learning Components
1. Data Preprocessing

Numerical scaling using MinMaxScaler

One-hot encoding for categorical fields

ColumnTransformer pipeline for reproducibility

ID columns added for Power BI integration

2. Dimensionality Reduction: PCA

PCA with 11 components

Variance and cumulative variance exported for visualization

Embedding space used for all clustering models

3. Clustering Models
DBSCAN

Detects dense clusters and flags anomalies where label = -1

Generates:

DBSCAN labels

DBSCAN identifier (0/1)

DBSCAN category (“OUTLIER”, “NOT OUTLIER”)

Exports pca_db_file.csv, cluster_db_scores.csv

OPTICS

Uses reachability and ordering to define structure

Generates:

Reachability plots

OPTICS labels and identifiers

4-level risk categories via quantiles

Exports optics_risk.csv, optics_score.csv

HDBSCAN

Hierarchical density clustering

Produces soft cluster membership probabilities

Generates:

Probabilities for each point

HDBSCAN labels

HDBSCAN identifier

Risk levels (“critical” → “stable”)

Exports hd_probability.csv, hd_scores.csv

Final Combined Risk Engine

The system produces a final fraud signal using all three models:

total identifier = DBSCAN + OPTICS + HDBSCAN
total category risk = {0: low, 1: medium, 2: high, 3: critical}


This creates a unified anomaly score for use in dashboards and workflows.

Power BI Dashboard Integration

A complete interactive dashboard is built to visualize:

Fraud severity levels

PCA variance + cumulative variance

DBSCAN clusters

OPTICS reachability

HDBSCAN probability distribution

Risk categories

Final fraud score

KPI cards tracking fraud volume and severity

Slicers enabling model & category filtering

Automated alerts for high-risk anomalies

Technologies Used

Python: pandas, numpy, sklearn, hdbscan

Machine Learning: PCA, DBSCAN, OPTICS, HDBSCAN

Visualization: Power BI

Data Engineering: Pipelines, ColumnTransformer

How to Run
python main_pipeline.py

Enter:

d → Run DBSCAN

o → Run OPTICS

h → Run HDBSCAN

f → Generate final fraud risk file

By:
Antonio Park
Retired U.S. Marine Corps Program Manager → Machine Learning Engineer
Focused on fraud detection, anomaly detection, automation, and dashboard analytics.
