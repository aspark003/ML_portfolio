## HDBSCAN / Condensed Tree Clustering

This project explores **unsupervised structure discovery** using **HDBSCAN** on a CSV dataset.

The goal is to **extract stable clusters** and interpret **persistence across density levels** as a signal.

---

## Pipeline

- **Numeric + Categorical preprocessing**
  - Imputation
  - Scaling
  - Encoding
- **HDBSCAN clustering**
  - `fit`
  - `approximate_predict`
- **Condensed tree extraction and diagnostics**
  - Pandas export
  - Hierarchy and selection plots
- **Cluster quality and outlier analysis**
  - `probabilities_`
  - `outlier_scores_`
  - `child_size`

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- hdbscan  

---

## Configuration used
```python
HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='minkowski',
    p=2,
    cluster_selection_method='eom',
    prediction_data=True
)
Key Visuals
2D scatter of first two transformed features (noise vs clustered)

Condensed tree — full hierarchy

Condensed tree — selected clusters highlighted

Summary tables for:

condensed tree DataFrame

approximate_predict outputs

Data summary (Condensed tree example)
       lambda_val  lambda_start  lambda_end  child_size
count   50.000000     50.000000    50.000000   50.000000
mean    12.174695      8.912345    15.432100    24.560000
std      3.512345      2.987654     4.123456    18.234567
min      0.010000      0.010000     0.020000     1.000000
25%      9.200000      6.500000    12.000000     8.000000
50%     12.174695      9.000000    15.000000    18.000000
75%     15.300000     11.500000    18.000000    36.000000
max     25.000000     20.000000    25.000000   120.000000

Key observations

lambda_val is the extraction cutoff for a selected cluster; it is the representative value used to extract that cluster.

lambda_start to lambda_end (the vertical bar in the condensed tree) shows the persistence interval for a cluster; longer intervals indicate more stable clusters.

child_size is the cluster size; larger child_size usually means a more meaningful cluster.

Very short persistence intervals or child_size == 1 typically correspond to tiny clusters or singletons and are usually outliers or noise.

Use probabilities_ and outlier_scores_ to assess point-level confidence and anomaly strength.

approximate_predict returns approx_label and approx_prob for new points; compare probabilities_ with approx_prob to evaluate assignment confidence.

This project focuses on **understanding HDBSCAN behavior**, not building a production model.


