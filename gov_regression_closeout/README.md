Financial Close-Out Exception Review Dashboard
Project Overview

This project implements a model-based financial close-out exception review system designed to support contract close-out, audit preparation, and spending verification workflows.

The system uses machine learning to learn expected spending behavior, measure deviations from that expectation, and rank records into review priority levels (High / Medium / Low).
Results are delivered through a Power BI dashboard for operational review and decision support.

This approach does not require labeled fraud data and is suitable for environments where verification, traceability, and audit defensibility are required.

Problem Statement

During financial close-out, large datasets must be reviewed to identify records that may require verification due to unexpected spending behavior.
Manual review is time-consuming and does not scale.

This project addresses that challenge by:

Learning what “normal” spending looks like given contextual variables

Identifying records that deviate from expected behavior

Prioritizing review effort based on relative deviation severity

Solution Approach
1. Data Ingestion & Preparation

Supports Excel, CSV, TSV, JSON, and TXT inputs

Performs structured cleaning, column normalization, and ID generation

Preserves raw and intermediate outputs for traceability

2. Model-Based Deviation Detection

Trains a Random Forest regression model on financial features

Uses the model to estimate expected funds used

Computes residuals (actual − predicted) for each record

3. Severity Ranking (Unsupervised by Intent)

Residuals are ranked using quantile-based thresholds

Records are labeled into:

High → highest priority for verification

Medium → secondary review

Low → normal / expected behavior

Direction of residuals indicates over-spend vs under-spend

No fraud labels or assumptions are used

4. Dashboard Delivery

Outputs are integrated into Power BI

Provides:

Count of records requiring review

Review level indicators

Row-level drill-down for verification

Designed for close-out and audit review, not accusation

Dashboard Purpose

The dashboard answers three operational questions:

How many records require review?

What is the review priority level?

Which specific records should be verified first?

It is intended as a decision-support and prioritization tool, not a compliance or enforcement system.

Key Outputs

actual funds used

predicted funds used

residual

residual label (1 / 2 / 3)

residual category (Low / Medium / High)

These outputs enable:

Verification prioritization

Close-out review support

Audit defensibility

Technologies Used

Programming & ML

Python

scikit-learn

Random Forest Regression

Quantile-based ranking

Data Engineering

pandas

ColumnTransformer

OneHotEncoder

MinMaxScaler

Pipeline-based preprocessing

Visualization

Power BI

Interactive tables and cards

Review-level filtering

FINAL

Eliminates manual scanning of large financial datasets

Preserves full audit traceability

Provides explainable, ranked review signals

Suitable for defense finance and government close-out workflows

Demonstrates regression-based anomaly detection used in an unsupervised manner
