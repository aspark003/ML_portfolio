https://app.fabric.microsoft.com/groups/a275037f-e2d6-4589-bae8-2443ddb165d7/dashboards/be597aa9-3975-4122-bbb5-2ddbda2dedaa?experience=fabric-developer
https://app.powerbi.com/groups/me/dashboards/be597aa9-3975-4122-bbb5-2ddbda2dedaa?experience=power-bi



Gov Finance Fraud Dashboard – FY 26
Project Overview

This project delivers a fully automated fraud detection and validation pipeline for government finance data. Using a combination of unsupervised anomaly detection and regression-based influence metrics, the system identifies potentially fraudulent transactions and ranks them by severity.

The solution integrates Python ML/AI processing with Power BI dashboards to provide interactive, real-time insights for stakeholders.

Features

Automated ML/AI Pipeline

Linear Regression, Ridge Regression, and Lasso Regression models applied to obligations data.

Cook’s distance calculated to detect influential points per model.

Fraud identifiers from DBSCAN, OPTICS, and HDBSCAN anomaly detectors integrated for cross-validation.

Combined regression final category (critical, high, medium, low, none) automatically computed per row.

Dynamic Data Processing

Fully automated ingestion, scoring, and aggregation of transactions.

Handling of edge cases including inf / NaN in Cook’s distance.

Row alignment preserved across all models and identifiers.

Power BI Dashboard Integration

Row-level table visual displaying:

Fraud identifiers

Cook’s distance scores and categories for Linear, Ridge, Lasso

Regression final identifier and category

Cards showing:

Total fraud identifiers

Total records

Fraud rate (percentage of flagged transactions)

Filters for anomaly type, budget authority, and other transactional fields.
File Structure
File	Description
gov_soft_gl_auto_dash_file4.csv	Raw obligations and transaction data
gov_regression1.csv	Linear Regression outputs with Cook’s distance scores
gov_regression2.csv	Ridge Regression outputs
gov_regression3.csv	Lasso Regression outputs
gov_regression4.csv	Final regression combined outputs with fraud identifiers and categories

Usage Instructions

Python Pipeline

Run ClusterRegression class in Python.

Use the following commands to execute each stage:

Linear Regression: model_name = 'l'

Ridge Regression: model_name = 'r'

Lasso Regression: model_name = 'la'

Final Regression Output: model_name = 'f'

CSV outputs will be generated automatically for each stage.

Power BI Dashboard

Open Power BI Desktop.

Load gov_regression4.csv.

Verify columns: fraud identifier, LINEAR / RIDGE / LASSO cooks distance identifier, Regression final category.

Create visuals:

Table → show all model outputs per row.

Cards → display total fraud count, total records, and fraud rate.

Key Metrics

Regression Final Category

critical → highest risk (all signals agree)

high → strong risk (three signals agree)

medium → moderate risk (two signals agree)

low → minor risk (one signal)

none → no risk detected

Fraud Rate

Automatically computed as:

Total Fraud Identifiers / Total IDs

Displayed in Power BI card


