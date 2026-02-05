# DBSCAN – Unsupervised Anomaly Detection

This project explores **unsupervised anomaly detection** using **DBSCAN** on a csv file.

The goal is to understand:
- DBSCAN identifies anomalies as *noise*
- DBSCAN(eps=0.5, min_samples=6, metric='minkowski', p=2, algorithm='auto', n_jobs=-1)
  - Higher then 0.5 merged all noise's to normal
  - No hugh difference updating min_samples
  - Manhattan / Euclidean = Not much difference
    
## What’s implemented

- Pandas / NumPy data handling
- Preprocessing pipeline:
  - numeric imputation + MinMax scaling
  - categorical imputation + one-hot encoding
- DBSCAN clustering (Minkowski distance)
- Matplotlib visualizations:
  - Label distribution
  - Points sizes
  - cluster sizes
  - Noise vs normal behavior
- Cluster validity metrics:
  - Silhouette = 0.6247198968642527
  - Calinski–Harabasz = 24636.995506993306
  - Davies–Bouldin = 0.9722291403292819

## Key observations
- DBSCAN produces **binary anomalies** (Noise vs Normal)
- Changing distance metrics reshapes density neighborhoods
- Increasing `eps` improves clustering scores but can absorb anomalies
- Good clustering ≠ good anomaly detection

This project focuses on **understanding DBSCAN behavior**, not building a production model.

