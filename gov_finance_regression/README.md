
https://app.powerbi.com/groups/me/dashboards/3becc95e-5b8a-43d8-86b6-85cc65d37c13?experience=power-bi

Residual-Based Financial Anomaly Detection
Overview

This project implements an end-to-end residual-based financial anomaly detection pipeline using supervised regression.
Rather than relying on unsupervised anomaly detection alone, the system builds an expectation model, measures deviations from that expectation, and classifies severity using distribution-based thresholds.

The output is a record-level anomaly dataset designed for direct use in Power BI dashboards and audit workflows.

Pipeline Architecture

The workflow is intentionally modular and executed in stages:

Raw File → Load → Clean → Model → Residuals → Severity Labels → Dashboard

1. File Loading (FileLoader)

Supports multiple file types:

Excel (.xlsx)

Text Files (CSV)

Features:

Validates Excel files as proper ZIP archives

Normalizes ingestion into Pandas DataFrames

Saves a standardized intermediate CSV (practice1.csv)

Purpose:

Ensure reliable ingestion regardless of source format.

2. Data Cleaning & Preparation (CleanFile)

Key steps:

Applies a custom header offset

Normalizes column names

Removes trailing non-data rows

Adds a stable record ID

Saves a preserved original snapshot for later reconciliation

Uses ColumnTransformer for:

Numeric median imputation

Categorical missing-value handling

Outputs:

practice_original.csv (pre-model reference)

practice0.csv (model-ready dataset)

Purpose:

Prepare a leakage-free, model-safe dataset while preserving raw financial context.

3. Expectation Modeling (LinearModel)

Models tested:

Linear Regression

LassoCV

RidgeCV

ElasticNetCV (available)

Evaluation strategy:

Train/Test split

Cross-validation (CV)

Full-dataset fit for expectation generation

Important design choice:

Cross-validation is used to validate realism, not maximize scores.

4. Leakage Control

Post-event and outcome-derived financial fields are explicitly excluded from modeling, including:

Commitments

Budget authority

Limits

Organizational rollups

These fields remain available for dashboard explanation, but not for prediction.

Purpose:

Prevent artificial performance inflation and ensure real-world validity.

5. Residual Calculation

Residuals are computed as:

residual = actual − predicted


This produces an interpretable deviation from expected spending.

6. Severity Classification

Residuals are categorized using quantile thresholds:

Under: ≤ 25th percentile

Normal: 25th–75th percentile

Over: ≥ 75th percentile

Outputs:

Numeric residual identifier

Human-readable severity label

Purpose:

Convert numeric deviation into actionable audit signals.

7. Output for Visualization

Final output is written back to practice_original.csv with:

Original financial context

Model predictions

Residual values

Severity categories

This file is designed for direct ingestion into Power BI.

Power BI Dashboard

The dashboard supports:

Record-level inspection

Severity filtering

Accounting-formatted financial fields

Residual-driven prioritization

Audit-ready interpretation

No model logic is duplicated in Power BI — all intelligence is upstream.

Why This Approach

Many financial anomaly systems fail due to:

Data leakage

Overfitting

Uninterpretable anomaly scores

This project prioritizes:

Model honesty

Explainability

Practical decision support

Residual-based anomalies provide a transparent and defensible signal suitable for financial review and governance contexts.

