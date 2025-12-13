Gov Finance Fraud Detection Dashboard
Project Overview

This project implements an end-to-end fraud detection system using unsupervised machine learning techniques. It identifies anomalous financial transactions and assigns fraud identifiers and risk categories (Critical, Medium, Low, None) for monitoring and review.

The system uses the following models:

PCA (Principal Component Analysis) – reduces dimensionality and highlights key features

DBSCAN – detects local density anomalies

OPTICS – identifies clusters of varying density

HDBSCAN – finds hierarchical clusters and outliers

Isolation Forest – isolates anomalies probabilistically

The dashboard includes the following measures:

Fraud Count – total number of transactions flagged as fraud

Total Records – total number of transactions in the dataset

Fraud Rate – proportion of transactions flagged as fraud

Features

Detects anomalies using PCA + DBSCAN, OPTICS, HDBSCAN, and Isolation Forest.

Assigns fraud identifiers and risk categories based on model agreement.

Generates a Power BI dashboard including:

Row-level audit table with all identifiers, detectors, and categories

Summary Cards for Fraud Count, Total Records, and Fraud Rate

Dataset context notes (FY, Account Code, Fund, Budget Line Item)

Supports alerts on Fraud Count or Fraud Rate for real-time monitoring of new fraud occurrences.

Folder Structure

anomaly_fraud/

gov_clean_cluster.csv — input raw dataset for processing

clusterF_pipeline.py — Python scripts that run the anomaly/fraud detection pipeline (DBSCAN, OPTICS, HDBSCAN, Isolation Forest, PCA)

gov_soft_gl_auto_dash_file4.csv — final processed dataset including fraud identifiers, anomaly categories, and computed scores

dashboard.pbix — Power BI dashboard showing row-level audit table, summary cards, and dataset context notes

README.md — this documentation file

Instructions

Python preprocessing

Run clusterF_pipeline.py to process new datasets.

Output is saved to gov_soft_gl_auto_dash_file4.csv.

Power BI

Open dashboard.pbix.

Refresh dataset to include new rows.

Cards and tables update automatically.

Monitoring

Set alerts on Fraud Count or Fraud Rate cards in Power BI to notify when thresholds are exceeded (e.g., when new fraud is detected).

Alerts trigger automatically whenever new rows are processed through the Python pipeline and meet the alert condition.

Notes

Do not modify the dataset schema; only add new rows. Columns and their order must remain unchanged.

All anomaly identifiers and fraud labels (DBSCAN, OPTICS, HDBSCAN, Isolation Forest) are computed automatically by the Python pipeline.

Dashboard visuals (table, cards, text boxes) are designed for auditability, clarity, and real-time monitoring.

