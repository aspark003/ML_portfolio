https://github.com/aspark003/ML_portfolio/blob/main/anomaly_fraud/README.md
https://app.powerbi.com/groups/me/dashboards/e510cda9-219c-4975-bf8c-15f8642f02bc?experience=power-bi

Gov Finance Fraud Detection Dashboard
Project Overview

Implements an end-to-end fraud detection system using unsupervised machine learning.

Identifies anomalous financial transactions and assigns fraud identifiers with risk categories (Critical, Medium, Low, None).

Models Used

PCA (Principal Component Analysis): reduces dimensionality and highlights key features

DBSCAN: detects local density anomalies

OPTICS: identifies clusters of varying density

HDBSCAN: finds hierarchical clusters and outliers

Isolation Forest: isolates anomalies probabilistically

Dashboard Measures

Fraud Count: total number of transactions flagged as fraud

Total Records: total number of transactions in the dataset

Fraud Rate: proportion of transactions flagged as fraud

Features

Detects anomalies using PCA + DBSCAN, OPTICS, HDBSCAN, and Isolation Forest.

Assigns fraud identifiers and risk categories based on model agreement.

Generates a Power BI dashboard with:

Row-level audit table including all identifiers, detectors, and categories

Summary cards for Fraud Count, Total Records, and Fraud Rate

Dataset context notes (FY, Account Code, Fund, Budget Line Item)

Alerts for Fraud Count or Fraud Rate for real-time monitoring of new fraud occurrences

Folder Structure

anomaly_fraud/

raw_data/

gov_clean_cluster.csv – input raw dataset for processing

processed_data/

gov_soft_gl_auto_dash_file4.csv – final dataset including fraud identifiers and scores

scripts/

clean.py – data cleaning and preprocessing

full_script.py – full anomaly/fraud detection pipeline

load_file.py – utility to load datasets into pipeline

docs/

dashboard.pbix – Power BI dashboard

powerbi_screenshot.pdf – dashboard screenshot for reference

README.md – this documentation file

Instructions
Python Preprocessing

Run full_script.py to process new datasets.

Output is automatically saved to gov_soft_gl_auto_dash_file4.csv.

Power BI

Open dashboard.pbix.

Refresh dataset to include new rows.

Cards and tables update automatically.

Monitoring & Alerts

Set alerts on Fraud Count or Fraud Rate cards to notify when thresholds are exceeded.

Alerts are triggered automatically whenever new rows are processed through the Python pipeline and meet the alert conditions.

Notes

Do not modify the dataset schema; only add new rows. Columns and their order must remain unchanged.

All anomaly identifiers and fraud labels (DBSCAN, OPTICS, HDBSCAN, Isolation Forest, PCA) are computed automatically.

Dashboard visuals (table, cards, text boxes) are designed for auditability, clarity, and real-time monitoring.
