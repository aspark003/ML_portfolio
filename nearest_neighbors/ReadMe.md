
## NEAREST NEIGHBORS / kneighbors 

This project explores **unsupervised structure discovery** using **nearest neighbors** on a CSV dataset.

The goal is to understand distance behavior as a signal.
- pipeline
  - Numeric + Categorical preprocessing (imputation, scaling, encoding)
  - Distance and index extraction (kneighbors)
  - Distance pair scatter diagnositics
  - Normalized statistical summaries

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- nearest neighbors

## Configuration used
```python
NearestNeighbors(
    n_neighbors=6,
    metric='minkowski',
    p=1,
    algorithm='auto',
    leaf_size=30
    random_state=42,
)
```
## Key Visuals
- Nearest neighbors distance measures
- Nearest neighbors index measures
- Metric comparison

## Data summary (Distance/Indices)
```
       Distance: 0   Distance: 1   Distance: 2   Distance: 3   Distance: 4   Distance: 5
count      32581.0  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000
mean           0.0      0.074130      0.088639      0.098019      0.086721      0.090165
std            0.0      0.060757      0.061892      0.064104      0.063052      0.063065
min            0.0      0.000000      0.000000      0.000000      0.000000      0.000000
25%            0.0      0.036163      0.050704      0.058475      0.048211      0.051072
50%            0.0      0.060433      0.074099      0.082033      0.070367      0.073692
75%            0.0      0.094511      0.107353      0.117319      0.105896      0.109478
max            0.0      1.000000      1.000000      1.000000      1.000000      1.000000

         Indices: 0    Indices: 1    Indices: 2    Indices: 3    Indices: 4    Indices: 5
count  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000
mean       0.499915      0.496047      0.491692      0.490091      0.487982      0.487911
std        0.288660      0.285050      0.284879      0.285108      0.285245      0.284481
min        0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
25%        0.249939      0.252072      0.244759      0.244796      0.241014      0.242679
50%        0.499693      0.492234      0.483287      0.479831      0.475429      0.472006
75%        0.749908      0.741789      0.737561      0.737306      0.734614      0.734207
max        1.000000      1.000000      1.000000      1.000000      1.000000      1.000000

```

## Key observations
- Distance: 0 acts as a baseline anchor (self_distance)
- Distance growth forms a stable geometric signal
- Neighbors tighten as it moves from distance to distance
- 
This project focuses on **nearest neighbors distance signal**, not building a production model.

