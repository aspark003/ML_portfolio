
## OPTICS / reachability-ordering 

This project explores **unsupervised structure density based structure discovery** using **reachability/ordering** on a CSV dataset.

The goal is to visualize and reaso.
- pipeline
  - Numeric + Categorical preprocessing (imputation, scaling, encoding)
  - Reachability extraction and normalization
  - Combined reachability / ordering signal matplot
  - Normalized statistical summaries

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- optics

## Configuration used
```python
OPITCS(
    min_samples=5,
    metric='minkowski',
    p=2,
    algorithm='auto'
)
```
## Key Visuals
- Green (Reachability)
  - Represents density distance - valleys indicate dense regions, spikes indicate sparse transition or outliers
- Red (Ordering)
  - Represents position - provides the coordinates system for the signal
  - Ordering surrounds reachability because ordering is index based, while reachability is geometry based

## Data summary (Reachability/Ordering)
```
       reachability      ordering
count  32581.000000  32581.000000
mean       0.029096      0.500000
std        0.025821      0.288688
min        0.000000      0.000000
25%        0.017212      0.250000
50%        0.023200      0.500000
75%        0.034503      0.750000
max        1.000000      1.000000

```

## Key observations
- Reachability is the true signal
- Ordering is the corrdinate system
- OPTICS structure emerges as valleys, not labels
- Tuning improves signal tightness before labels matter
- The algorithm is behaving exactly as designed
  
This project focuses on **understanding OPTICS behavior**, not building a production model.


