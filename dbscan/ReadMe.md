-----DBSCAN – Unsupervised - Noise / Cluster

This project explores **unsupervised anomaly detection** using **DBSCAN** on a CSV dataset.

The goal is to understand how DBSCAN behaves with respect to:
- identifying anomalies as *noise*
- sensitivity to `eps`, `min_samples`, and distance metrics

----- Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib

----- Configuration used
--------------PYTHON---------------
DBSCAN(eps=0.5,min_samples=6,metric='minkowski',p=2,algorithm='auto',n_jobs=-1)
    
----- What’s implemented

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
  - Silhouette = 0.6244500631237574
  - Calinski–Harabasz = 24665.92423587026
  - Davies–Bouldin = 0.5126475719931072

----- Data summary (derived)

              Label         Count 
count  32581.000000  32581.000000 
mean       2.917805      0.326484 
std        2.445062      0.203755 
min       -1.000000      0.000000 
25%        1.000000      0.166667 
50%        2.000000      0.250000 
75%        4.000000      0.416667 
max       11.000000      1.000000 

  
----- Key observations

 -DBSCAN produces binary anomaly output (noise vs cluster)
 -Distance metrics reshape density neighborhoods but may not change outcomes significantly
 -Increasing eps improves clustering scores while reducing detected anomalies
 -Strong clustering performance can conflict with anomaly detection goals

------- This project focuses on understanding DBSCAN behavior, not building a production model -------

