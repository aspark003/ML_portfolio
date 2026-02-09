Multi-Model Risk Detection – Unsupervised – Density & Decision Fusion

This project explores unsupervised anomaly detection and risk scoring on a CSV dataset using a combination of density-based and decision-based models, followed by score normalization and fusion.

The goal is to:

detect outliers / rare observations without labels

compare how different unsupervised models behave

combine model outputs into a single interpretable risk signal

Models Used

Density-based

DBSCAN

OPTICS

HDBSCAN

Decision-based

Local Outlier Factor (LOF)

Isolation Forest

One-Class SVM

Dependencies

pandas

numpy

scikit-learn

hdbscan

Configuration Used
Dimensionality Reduction
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

What’s Implemented
Data Loading

CSV ingestion using pandas

Original dataframe preserved for interpretation

Preprocessing Pipeline

Numerical features

Median imputation

Min-Max scaling

Categorical features

Most-frequent imputation

One-Hot Encoding (drop='first', unknown-safe)

Implemented using:

Pipeline

ColumnTransformer

Feature Space Preparation

Preprocessing outputs a dense feature matrix

PCA reduces dimensionality while retaining 90% variance

All models operate in PCA space

Model Outputs

Each model produces:

a raw label (cluster / inlier / outlier)

a continuous severity score scaled to [0, 1]

a severity level based on quantiles:

Low

Medium

High

Critical (for fused outputs)

Fusion Logic
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

Density anomaly score

Decision severity score

Final outputs:

risk detection score

risk detection level

Severity Thresholding

Severity levels are quantile-based, not fixed thresholds:

Medium ≈ 60th percentile

High ≈ 85th percentile

Critical ≈ 95th percentile

This makes the system:

adaptive to dataset size

robust to scale differences

usable across domains

Output Summary (Derived)

The final dataframe includes:

cluster labels (DBSCAN / OPTICS / HDBSCAN)

individual model severity scores & levels

density fusion scores

decision fusion scores

final risk detection score

final risk detection level

sequential id for traceability

Key Observations

No single unsupervised model is sufficient on its own

Density-based models capture local structure

Decision-based models capture global deviation

Fusion stabilizes noisy individual scores

Quantile-based severity improves interpretability

High clustering quality does not always imply good anomaly detection

Scope & Intent

This project focuses on:

understanding unsupervised anomaly behavior

comparing model perspectives

building a diagnostic risk signal
