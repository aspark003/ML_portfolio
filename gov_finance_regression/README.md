https://app.powerbi.com/groups/me/dashboards/3becc95e-5b8a-43d8-86b6-85cc65d37c13?experience=power-bi

Residual-Based Financial Deviation & Risk Screening Overview

This project implements an end-to-end regression-driven deviation analysis pipeline for financial and ledger-style data.
Rather than treating regression as a prediction task, the system builds an expectation model, measures record-level deviations, assigns severity, and flags items for investigative review.

The goal is not forecasting accuracy, but quantifying material deviation and supporting audit, compliance, and risk prioritization workflows.

The output is a record-level enriched dataset designed for direct use in Power BI dashboards and human review processes.

Pipeline Architecture

The workflow is modular and executed in clearly separated stages:

Raw File → Load → Clean → Model → Residuals → Severity → Investigation → Dashboard

File Loading (FileLoader)

Supported formats:

Excel (.xlsx)

Text files (.csv, .tsv)

JSON / TXT

Features:

Validates Excel files as proper ZIP archives

Normalizes ingestion into Pandas DataFrames

Saves a standardized intermediate file (practice1.csv)

Purpose:
Ensure reliable ingestion regardless of source format while preserving raw structure.

Data Cleaning & Preparation (CleanFile)

Key steps:

Applies custom header offsets where required

Normalizes column names

Removes trailing non-data rows

Adds a stable record ID

Preserves an untouched original snapshot for reconciliation

Uses ColumnTransformer for:

Numeric median imputation

Categorical missing-value handling

Feature-safe preprocessing

Outputs:

practice_original.csv (audit reference)

practice0.csv (model-ready dataset)

Purpose:
Prepare a leakage-controlled dataset while preserving full financial context for explanation.

Expectation Modeling (LinearModel)

Regression is used only to establish an expected value, not to classify records.

Models evaluated:

Linear Regression

LassoCV

RidgeCV

ElasticNetCV (available)

Evaluation approach:

Train/Test split

Cross-validation for realism checks

Full-dataset fit for expectation generation

Design principle:
Model evaluation validates plausibility — it does not drive downstream decisions.

Leakage Control

To prevent artificial performance inflation, outcome-derived or post-event fields are excluded from modeling, including:

Commitments

Budget authority

Limits

Organizational rollups

These fields remain available for dashboard interpretation, but are never used for prediction.

Purpose:
Ensure real-world validity and audit defensibility.

Residual Calculation

Residuals are computed per record as:

residual = actual − predicted

This produces a direct, interpretable measure of financial deviation from expectation.

Residuals are not scores — they are differences.

Severity Classification

Residuals are translated into severity using distribution-based thresholds:

Under → ≤ 25th percentile

Normal → 25th–75th percentile

Over → ≥ 75th percentile

Outputs:

Numeric residual identifier

Human-readable residual category

Binary investigation flag

Purpose:
Convert numeric deviation into actionable audit signals.

Investigation Logic

Residual severity is collapsed into a binary review gate:

Investigation required

None

This mirrors the same decision structure used in the unsupervised anomaly pipeline.

Regression ends here — all further work is analysis, not modeling.

Output for Visualization

Final output is written back to practice_original.csv and includes:

Original financial context

Model predictions

Residual values

Severity labels

Investigation flags

The dataset is Power BI–ready with no downstream model logic.

Power BI Dashboard

The dashboard supports:

Record-level inspection

Residual severity filtering

Accounting-formatted financial fields

Deviation-driven prioritization

Audit-ready interpretation

No ML logic is duplicated in Power BI — all intelligence is upstream.

Why This Approach

Many financial anomaly systems fail due to:

Data leakage

Overfitting

Opaque anomaly scores

Undocumented decision logic

This project prioritizes:

Model honesty

Interpretability

Governance defensibility

Practical decision support

Residual-based deviation analysis provides a transparent, explainable, and defensible signal suitable for financial review, audit, and compliance contexts.
