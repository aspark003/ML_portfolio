# DBSCAN – Unsupervised – Noise / Clusters

This project explores **unsupervised clustering** (and **noise detection**) using **DBSCAN** on a CSV dataset.

The goal is to understand how DBSCAN behaves with respect to:
- identifying **noise points** (`label = -1`)
- sensitivity to `eps`, `min_samples`, and distance metric choice

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib

## Configuration used
```python
DBSCAN(eps=0.5, min_samples=6,metric='minkowski',p=2,algorithm='auto', n_jobs=-1)
What’s implemented
Pandas/NumPy data loading and cleanup

Preprocessing pipeline:

numeric: SimpleImputer(mean + indicator) + MinMaxScaler

categorical: SimpleImputer(constant) + OneHotEncoder

DBSCAN fit on the transformed feature matrix

Visualizations:

cluster label - number of points (noise vs clusters)

cluster size distribution (how many clusters have the same size)

Cluster validity metrics:

silhouette_score: 0.6244500631237574

calinski_harabasz_score: 24665.92423587026

davies_bouldin_score: 0.5126475719931072

Key Visuals
Label counts (cluster id vs points; -1 highlighted as noise)

Cluster-size distribution (frequency of cluster sizes)

Data summary (derived)
              Label  Label scaled       Points  Points scaled  Cluster  Cluster Scaled
count  32581.000000  32581.000000    13.000000      13.000000     13.0            13.0
mean       2.917805      0.326484  2506.230769       0.428025      1.0             0.0
std        2.445062      0.203755  2391.676718       0.408624      0.0             0.0
min       -1.000000      0.000000     1.000000       0.000000      1.0             0.0
25%        1.000000      0.166667   509.000000       0.086793      1.0             0.0
50%        2.000000      0.250000   620.000000       0.105758      1.0             0.0
75%        4.000000      0.416667  5014.000000       0.856484      1.0             0.0
max       11.000000      1.000000  5854.000000       1.000000      1.0             0.0
Key observations
DBSCAN outputs cluster labels plus a noise label (-1).

Noise detection is not binary overall; it is multi-cluster + noise.

Increasing eps typically reduces noise and merges clusters.

Increasing min_samples typically increases noise and removes small clusters.

Strong clustering metrics can conflict with the goal of detecting many anomalies.

This project focuses on observing DBSCAN behavior and diagnostics, not building a production model.
