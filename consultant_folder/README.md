CDbscan: Multi-Model Unsupervised Fraud & Risk Detection Pipeline

This folder contains the full Python implementation of the CDbscan class.
The pipeline performs anomaly detection using DBSCAN, OPTICS, and HDBSCAN after data preprocessing and PCA dimensionality reduction.
All model outputs are exported as CSV files for Power BI dashboards and analysis.

Overview

The workflow:

Loads and prepares the dataset.

Removes non-predictive or leakage columns.

Normalizes numeric fields and encodes categorical fields.

Applies PCA with 11 components.

Fits DBSCAN, OPTICS, and HDBSCAN models.

Generates labels, identifiers, and risk categories for each model.

Exports full results to CSV for downstream reporting.

Produces a final combined risk score using all three models.

Removed Columns

The following columns are dropped before modeling:

urgency flag

geo distance to vendor

invoice match score

risk category

holiday period

is fraud

These fields either leak target information or are not predictive for unsupervised modeling.

Preprocessing Steps

MinMaxScaler scales all numeric variables.

OneHotEncoder encodes categorical variables.

ColumnTransformer merges numeric and categorical pipelines.

PCA (11 components) reduces dimensionality.

PCA output is then passed to DBSCAN, OPTICS, and HDBSCAN pipelines.

Model Outputs (Saved to OneDrive)
1. DBSCAN Outputs

var_cumsum.csv — variance and cumulative explained variance

pca_db_file.csv — PCA components + DBSCAN labels and identifiers

final.csv — original data with DBSCAN fields appended

cluster_db_scores.csv — Silhouette, Calinski, and Davies metrics

2. OPTICS Outputs

optics_risk.csv — reachability, ordering, labels, identifiers, risk categories

optics_score.csv — OPTICS model evaluation metrics

final.csv — updated with OPTICS labels and categories

3. HDBSCAN Outputs

hd_probability.csv — probability scores + probability risk bins

hd_scores.csv — evaluation metrics for HDBSCAN

final.csv — updated with HDBSCAN labels, identifiers, and categories

4. Final Combined Risk (final_db)

A combined risk score is created using the sum of all anomaly identifiers:

total identifier = DBSCAN + OPTICS + HDBSCAN


Risk mapping:

3 → critical

2 → high

1 → medium

0 → low

Final results are saved back into final.csv.

How to Run the Script

When executed, the script prompts:

Enter model name here:


Valid inputs:

d → run DBSCAN

o → run OPTICS

h → run HDBSCAN

f → generate the final combined risk file

Constructor Format
cd = CDbscan('c:/Users/anton/OneDrive/park_consultant.csv', model_name)

Notes

All files export automatically to your OneDrive directory.

PCA uses 11 components for consistency across all models.

Outliers are flagged when labels = −1.

Final combined risk is produced only when model name = f.
