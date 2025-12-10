# Consultant Folder

This folder contains consulting-related dashboards, scripts, and documentation for portfolio and project use.
CDbscan: Multi-Model Unsupervised Fraud and Risk Detection Pipeline

This directory contains the implementation of the CDbscan class, an end-to-end unsupervised machine learning workflow designed to identify anomalies and classify operational risk using three clustering algorithms: DBSCAN, OPTICS, and HDBSCAN. The pipeline standardizes the dataset, applies dimensionality reduction, executes each model, and exports structured outputs suitable for advanced analytics and Power BI dashboards.

Overview

The workflow performs the following steps:

Loads and cleans the dataset by removing non-predictive columns.

Scales numeric variables and encodes categorical variables.

Transforms the dataset using PCA with 11 components.

Fits DBSCAN, OPTICS, and HDBSCAN models for anomaly detection.

Generates risk labels, identifiers, and supporting evaluation metrics.

Exports all outputs to CSV files for reporting and dashboard integration.

Combines model results to produce a final risk category.

This design supports fraud detection, operational risk monitoring, and data-driven decision making.

Removed Non-Predictive Columns

The following fields are excluded to avoid leakage and maintain unsupervised learning integrity:

urgency flag
geo distance to vendor
invoice match score
risk category
holiday period
is fraud

Preprocessing Steps

The preprocessing architecture uses a consistent framework for all three clustering models:

Numeric features are normalized with MinMaxScaler.

Categorical features are encoded using OneHotEncoder.

PCA reduces dimensionality while retaining key variance.

Each clustering model uses identical preprocessed inputs for consistent comparison.

Model Output Files
DBSCAN (pc_db)

var_cumsum.csv
pca_db_file.csv
cluster_db_scores.csv
final.csv updated with DBSCAN labels, identifiers, and categories

OPTICS (pc_op)

optics_risk.csv
optics_score.csv
final.csv updated with OPTICS labels, identifiers, and risk categories

HDBSCAN (pc_hd)

hd_probability.csv
hd_scores.csv
final.csv updated with HDBSCAN labels, probability scores, and risk categories

Combined Risk Output (final_db)

The final step aggregates DBSCAN, OPTICS, and HDBSCAN results to compute a total risk identifier and a mapped risk category:

0 = low
1 = medium
2 = high
3 = critical

These combined classifications are written back into final.csv.

How to Run the Script

Run the file and enter a model selection when prompted.

Model options:

d runs DBSCAN
o runs OPTICS
h runs HDBSCAN
f runs the combined risk model

Example:

Enter model name here: d

Constructor example:

cd = CDbscan('c:/Users/anton/OneDrive/park_consultant.csv', model_name)

Notes

All outputs are saved to the OneDrive directory.
Outliers are identified when model labels equal minus one.
PCA with 11 components ensures consistency across DBSCAN, OPTICS, and HDBSCAN.
The combined risk score reflects agreement across all three models.

Power BI Dashboards

This pipeline is paired with a full suite of Power BI dashboards that visualize the detection results and provide an enterprise-grade monitoring interface.

Included dashboards:

DBSCAN and PCA Dashboard
Displays DBSCAN clusters, PCA projections, explained variance, and model evaluation scores.

OPTICS Reachability Dashboard
Shows reachability plots, ordering structures, OPTICS cluster labels, and risk segmentation.

HDBSCAN Probability Dashboard
Visualizes probability-based risk categories, label distributions, and cluster stability metrics.

Final Anomaly Summary Dashboard
Aggregates all three models into a unified operational view.
Shows total anomalies, category counts, and model score comparisons across DBSCAN, OPTICS, and HDBSCAN.

These dashboards are designed for fraud detection teams, financial auditors, program managers, and operational oversight groups.
