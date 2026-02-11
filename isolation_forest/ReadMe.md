Isolation Forest – Unsupervised Anomaly Detection

This project explores unsupervised anomaly detection using Isolation Forest on a mixed-type credit dataset.

The focus is on:

Understanding how anomaly scores are generated

Interpreting decision function behavior

Verifying separation between normal observations and noise

Analyzing score distribution structure

Overview

Isolation Forest isolates anomalies through random recursive partitioning.

Key ideas:

Anomalies require fewer splits to isolate.

Shorter average path length → more anomalous.

The decision_function() shifts scores so:

score < 0 → anomaly

score ≥ 0 → normal

Pipeline
Data Preprocessing

Numeric features

Median imputation (+ missing indicator)

Min–Max scaling

Categorical features

Constant-value imputation ("missing")

One-hot encoding (handle_unknown='ignore')

Implemented via ColumnTransformer and Pipeline for reproducibility.

Anomaly Detection (Isolation Forest)

Configuration used:

IsolationForest(
    n_estimators=200,
    max_samples=10,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)


Key outputs:

fit_predict() → anomaly labels

decision_function() → anomaly scores

Diagnostics & Analysis

The following diagnostics are produced:

1️⃣ Decision Function Separation

Scatter of sorted labels vs sorted decision scores.

Purpose:

Verify separation between normal and anomalous regions.

Confirm threshold at 0 is working.

Observe score range and spread.

2️⃣ Score Frequency Distribution

Unique decision values and their frequency.

Purpose:

Understand score concentration.

Identify central dense region.

Examine tail sparsity (anomalies).

Results
Dataset Summary

Total samples: 32,581

Contamination: 10%

Labels:

1 → normal

-1 → anomaly

Decision Function Summary
mean decision score: 0.032961
std: 0.024169
min: -0.079209
max: 0.089961


Interpretation:

Majority of scores are positive.

Anomalies appear in the negative tail.

Clear numeric separation around 0.

Score Distribution
unique decision values: 29,331
mean frequency: ~1.11
max frequency: 10


Interpretation:

Most decision scores are unique.

Isolation Forest produces near-continuous scoring.

Distribution shows dense central region and sparse lower tail.

Interpretation Notes

The threshold at 0 separates normal from anomalous observations.

max_samples=10 leads to high randomness and near-unique decision values.

Score distribution is right-shifted (dominant normal region).

Negative tail represents strongest anomalies.

Dependencies

pandas

numpy

scikit-learn

matplotlib

Scope

This project is exploratory.

The goal is to:

Understand Isolation Forest mechanics.

Validate anomaly score behavior.

Visualize score structure and separation.

Build intuition around tree-based isolation.

No hyperparameter tuning or production deployment is performed.
