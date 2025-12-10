Government Invoice Anomaly Detection System

This folder contains the full anomaly-detection pipeline built for government invoice datasets, including Python scripts, PCA outputs, and Power BI dashboards.

Files in This Folder
Python Script

gov_script.py — Full PCA + DBSCAN + OPTICS + HDBSCAN pipeline.

Power BI Dashboards

gov_pca_dbscan.pbix — PCA + DBSCAN visualization and outlier detection.

gov_reachability_optics_updated.pbix — OPTICS reachability and ordering visualization.

gov_pca_hdbscan_prob.pbix — HDBSCAN probability visualization and risk scoring.

gov_final_updated.pbix — Final combined risk dashboard (DBSCAN + OPTICS + HDBSCAN).

What the Pipeline Does
1. Preprocessing

Cleans column names.

Removes date fields.

Handles numeric/categorical missing values.

Scales numeric features.

One-hot encodes categorical features.

2. PCA (Principal Component Analysis)

Reduces dimensionality → improves clustering.

Uses 4 components for all models.

Exports variance + cumulative variance.

3. Clustering Models

DBSCAN → density-based outliers.

OPTICS → reachability + ordering structure.

HDBSCAN → probability-based risk scoring.

Each model exports:

labels

identifiers (1 = outlier)

risk category

PCA coordinates

4. Final Combined Risk

Adds:
DBSCAN identifier  
OPTICS identifier  
HDBSCAN identifier  

total identifier  
total category (critical, high, medium, low)
Exported Output Files (CSV)

gov_variance_cumsum.csv

gov_pca_dbscan.csv

gov_optics_reachability.csv

gov_optics_scores.csv

gov_hdbscan_probability.csv

gov_hdbscan_score.csv

gov_cluster_final.csv (master file used for final dashboard)

How to Run the Script

Run:
python gov_script.py
d → DBSCAN  
o → OPTICS  
h → HDBSCAN  
f → Final combined risk  
Example:
Enter model name here: d
Power BI Usage

Each .pbix file is connected to its matching CSV output.
These dashboards include:

PCA visuals

Reachability plots

Probability distributions

Outlier tables

Slicers for full invoice exploration

The final dashboard (gov_final_updated.pbix) contains the combined risk score and full anomaly summary.







