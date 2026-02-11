## HDBSCAN / Condensed Tree Clustering

This project explores **unsupervised structure discovery** using **HDBSCAN** on a mixed-type CSV dataset.

The focus is on **extracting stable clusters**, understanding **cluster persistence across density levels**, and using **probability and outlier signals** to reason about structure rather than to build a production-ready model.

---

## Overview

HDBSCAN is applied after a full preprocessing pipeline (numeric + categorical). The analysis emphasizes:

* Stability of clusters via the **condensed tree**
* Separation between **core clusters and noise**
* Point-level confidence using **membership probabilities** and **outlier scores**
* How HDBSCAN behaves under different density assumptions

---

## Pipeline

### Data preprocessing

* **Numeric features**

  * Mean imputation (+ missing indicator)
  * Min–Max scaling
* **Categorical features**

  * Constant-value imputation (`"missing"`)
  * One-hot encoding (`handle_unknown='ignore'`)

Implemented using `ColumnTransformer` and `Pipeline` for reproducibility.

---

### Clustering (HDBSCAN)

Configuration used:

```python
HDBSCAN(min_cluster_size=2,min_samples=15,metric='minkowski',p=1)
```

Key steps:

* `fit_predict` on the transformed feature matrix
* Extraction of:

  * `labels_`
  * `probabilities_`
  * `outlier_scores_`
  * `cluster_persistence_`

---

### Diagnostics & Analysis

The following diagnostics are produced:

* **Cluster vs noise size diagnostic**

  * Distribution of cluster sizes
  * Separation between noise points (`label == -1`) and clusters

* **Membership probability analysis**

  * Point-level confidence for cluster assignment
  * Noise points consistently show near-zero probability

* **Outlier score analysis**

  * Higher values indicate stronger anomaly behavior
  * Useful for ranking suspicious or unstable points

* **Sorted probability vs outlier comparison**

  * Visual contrast between confidence and anomaly strength

* **Cluster persistence stability**

  * Persistence values plotted per cluster
  * Highlights which clusters survive across density levels

---

## Key Visuals

* 2D scatter of cluster size vs cluster index (noise vs cluster)
* Membership probabilities across all data points
* Outlier scores across all data points
* Probabilities vs outlier scores (sorted)
* Cluster persistence stability plot

(See `hdbscan_plot.pdf` for rendered figures.)

---

## Metrics (for reference)

Although HDBSCAN is density-based and does not optimize these metrics directly, standard clustering scores are reported for context:

* Silhouette score
* Calinski–Harabasz score
* Davies–Bouldin score

These should be interpreted cautiously, especially in the presence of noise labels.

---

## Example Summary Statistics

### Point-level signals

* **probabilities_**

  * Mean close to 1.0 - strong cluster confidence for most points
* **outlier_scores_**

  * Heavy-tailed distribution → small set of strong anomalies

### Cluster-level signals

* **cluster_persistence_**

  * Higher persistence → more stable and meaningful clusters
  * Zero or near-zero persistence - weak or transient clusters

---

## Interpretation Notes

* `lambda_val` represents the cutoff at which a cluster is selected.
* `lambda_start` → `lambda_end` defines the persistence interval in the condensed tree.
* Longer persistence intervals generally indicate **more reliable structure**.
* Very small clusters or `child_size == 1` often correspond to noise or outliers.
* `approximate_predict` can be used for new data points; comparing `approx_prob` with training `probabilities_` helps assess assignment confidence.

---

## Dependencies

* pandas
* numpy
* scikit-learn
* matplotlib
* hdbscan

---

## Results

### Clustering outcome

* Total samples: **32,581**
* Noise points (`label == -1`): present and clearly separated by probability and outlier score
* Number of clusters: **16** (including weak / transient clusters)

### Metrics (observed)

```text
silhouette_score:0.485035460334384
calinski_harabasz_score:17602.790847669392
davies_bouldin_score:0.9739470072729963
```

These values suggest:

* Moderate separation between dense regions
* Compact, well-defined cores for dominant clusters
* Presence of noise and density variation, which is expected and desired under HDBSCAN

---

### Point-level diagnostics

**Membership probabilities (`probabilities_`)** and **outlier scores (`outlier_scores_`)** summary:

```text
       probabilities  outlier scores
count   32581.000000    32581.000000
mean        0.958247        0.057363
std         0.146939        0.102019
min         0.000000        0.000000
25%         1.000000        0.005805
50%         1.000000        0.017922
75%         1.000000        0.054738
max         1.000000        0.895384
```

Interpretation:

* Probabilities are heavily concentrated at **1.0**, indicating high-confidence assignments for most clustered points.
* Noise points sit near **0.0** probability (consistent with the probabilities plot).
* Outlier scores are **right-skewed**, with a small number of strong anomalies.

---

### Cluster-level diagnostics

**Cluster persistence (`cluster_persistence_`)** summary:

```text
count    16.000000
mean      0.140335
std       0.099752
min       0.000000
25%       0.082249
50%       0.133590
75%       0.215127
max       0.291290
```

Interpretation:

* High-persistence clusters represent stable structure across density levels
* Near-zero persistence clusters are fragile and likely density artifacts

---

### Visual confirmation

The rendered figures in `hdbscan_plot.pdf` confirm:

* Clear separation between cluster cores and noise
* Strong inverse relationship between probability and outlier score
* A small number of dominant, stable clusters
* Multiple weak clusters that collapse quickly in the condensed tree

---

## Scope

This project is intentionally **exploratory**.

The goal is to **understand HDBSCAN behavior and diagnostics**, interpret stability and uncertainty signals, and build intuition around density-based clustering — not a production model.
