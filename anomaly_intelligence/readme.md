# Multi-Model Risk Detection – Unsupervised – Production Pipeline

This project implements a **production-ready, fully unsupervised risk detection pipeline** for tabular data.  
It combines **density-based** and **decision-based** anomaly detection models, applies **robust preprocessing**, and produces **normalized, interpretable risk scores and severity levels** suitable for downstream systems.

The system is designed to:
- operate without labeled data
- handle mixed numerical and categorical features
- scale across datasets with different distributions
- provide stable, explainable risk signals

---

## Models Used

**Density-based**
- DBSCAN
- OPTICS
- HDBSCAN

**Decision-based**
- Local Outlier Factor (LOF)
- Isolation Forest
- One-Class SVM

---

## Dependencies
- pandas
- numpy
- scikit-learn
- hdbscan

---

## Configuration Used

### Dimensionality Reduction
```python
PCA(n_components=0.9, svd_solver='auto', random_state=42)
DBSCAN
DBSCAN(eps=0.2, min_samples=7, metric='euclidean', n_jobs=-1)
OPTICS
OPTICS(min_samples=3, xi=0.05, metric='euclidean')
HDBSCAN
HDBSCAN(
    min_samples=8,
    min_cluster_size=7,
    cluster_selection_method='eom',
    metric='euclidean'
)
Local Outlier Factor
LocalOutlierFactor(n_neighbors=100, metric='euclidean')
Isolation Forest
IsolationForest(n_estimators=100, random_state=42)
One-Class SVM
OneClassSVM(kernel='rbf', gamma='scale')
System Architecture
1. Data Ingestion
CSV-based batch ingestion

Encoding-safe read (utf-8-sig)

Original feature space preserved for auditability

2. Preprocessing Pipeline
Numerical features

Median imputation (robust to outliers)

Min-Max scaling

Categorical features

Most-frequent imputation

One-Hot Encoding with unknown handling

Implemented using:

Pipeline

ColumnTransformer

This ensures consistent transformations between training and inference.

3. Feature Space Management
Preprocessing produces a dense feature matrix

PCA retains 90% of explained variance

Reduces noise and improves model stability

All models operate in the same transformed space

Detection & Scoring
Individual Model Outputs
Each model produces:

raw labels (cluster / inlier / outlier)

continuous severity scores normalized to [0, 1]

severity levels: Low, Medium, High, Critical

Fusion Strategy
Density Anomaly Fusion
Average of:

DBSCAN severity score

OPTICS reachability severity score

HDBSCAN outlier severity score

Output:

density anomaly score

density severity level

Decision-Level Fusion
Average of:

LOF severity score

Isolation Forest severity score

One-Class SVM severity score

Output:

decision severity score

decision severity level

Final Risk Detection Fusion
Average of:

density anomaly score

decision severity score

Final outputs:

risk detection score

risk detection level

Severity Calibration
Severity levels are quantile-based and recalculated per dataset:

Medium ≈ 60th percentile

High ≈ 85th percentile

Critical ≈ 95th percentile

This allows:

automatic calibration across domains

stability under distribution shift

consistent alert volumes

Output Schema (Core Fields)
id

model labels (DBSCAN / OPTICS / HDBSCAN / LOF / Isolation / SVM)

individual severity scores & levels

density anomaly score

decision severity score

risk detection score

risk detection level

All outputs are deterministic given the same input data and configuration.

Operational Characteristics
Fully unsupervised

No dependency on historical labels

Robust to missing values

Handles mixed data types

Parallelized where supported

Deterministic preprocessing and scoring

Deployment Notes
Suitable for batch processing

Can be scheduled (cron / Airflow / Prefect)

Output integrates cleanly with:

rule engines

dashboards

alerting systems

downstream ML models

Scope
This system is intended for production risk detection, including:

credit risk screening

fraud pre-filtering

operational anomaly monitoring

compliance and audit pipelines

Summary
This pipeline delivers:

stable anomaly detection

multi-model consensus scoring

interpretable severity levels

production-grade preprocessing and fusion

It is designed to be deployed, monitored, and extended, not just explored.
