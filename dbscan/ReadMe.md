DBSCAN – Unsupervised – Noise / Clusters

This project explores unsupervised clustering and noise detection using DBSCAN on a mixed-type CSV dataset.

The goal is to understand how DBSCAN behaves with respect to:

identifying noise points (label = -1)

forming connected components under a fixed distance scale

sensitivity to eps, min_samples, and distance metric choice

The emphasis is diagnostic and behavioral, not production modeling.

Pipeline
Data preprocessing

Numeric features

Mean imputation (+ missing indicator)

Min–Max scaling

Categorical features

Constant-value imputation ("missing")

One-hot encoding (handle_unknown='ignore')

Implemented using Pipeline and ColumnTransformer for reproducibility.

Clustering (DBSCAN)

Configuration used:

DBSCAN(eps=0.5,min_samples=6,metric='minkowski',p=2,algorithm='auto',n_jobs=-1)

Key characteristics:

Single-scale, density-connected clustering

Hard assignments only (cluster ID or noise)

No hierarchy, persistence, or confidence scores

Diagnostics & Analysis
Cluster size distribution

A scatter plot of cluster label (sorted) vs cluster size highlights:

A single noise label (-1) with small cardinality

Several large connected components

Multiple small clusters that are equally valid under DBSCAN

This plot represents the entire structural output of DBSCAN at the chosen eps.

Cluster validity metrics (context only)
silhouette_score:        0.6232959411289837
calinski_harabasz_score: 4883.568224775606
davies_bouldin_score:     0.8110642656289275

Interpretation:

Clusters are geometrically compact and reasonably separated

Metrics ignore density stability and noise behavior

Scores are conditional on the chosen eps and sample

Label distribution summary

Summary statistics over cluster labels:

count    6516.000000
mean        4.380909
std         2.666140
min        -1.000000
25%         1.000000
50%         5.000000
75%         6.000000
max        11.000000

Notes:

Label values are categorical identifiers, not ordinal quantities

Mean and standard deviation have no semantic meaning

Presence of -1 confirms noise detection

Maximum label indicates 12 clusters at this scale

Key Observations

DBSCAN produces multi-cluster + noise, not binary noise detection

All clusters are treated as equally valid once formed

Increasing eps merges clusters and reduces noise

Increasing min_samples increases noise and removes small clusters

Strong geometric metrics can conflict with anomaly-detection goals

Dependencies

pandas

numpy

scikit-learn

matplotlib

Scope

This project focuses on observing DBSCAN behavior and diagnostics.
