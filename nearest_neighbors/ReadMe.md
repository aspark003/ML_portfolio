## NEAREST NEIGHBORS / kneighbors

This project explores **unsupervised structure discovery** using **nearest neighbors** on a CSV dataset.

The goal is to understand **distance behavior as a signal**.

---

## Pipeline

- **Numeric + Categorical preprocessing**
  - SimpleImputer (mean + indicator)
  - MinMaxScaler
  - OneHotEncoder
- **NearestNeighbors**
  - `fit`
  - `kneighbors()` extraction
- **Diagnostics**
  - Distance pair scatter (distance vs distance)
  - Normalized statistical summaries for distances and neighbor indices

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  

---

## Configuration used
```python
NearestNeighbors(
    n_neighbors=6,
    metric='minkowski',
    p=1,
    algorithm='auto',
    leaf_size=30,
    n_jobs=-1
)
Key Visuals
Nearest neighbors distance measures (pairwise distance scatter)

Nearest neighbors index measures (neighbor index scatter)

Normalized distance and index summary tables

Data summary (Distance / Indices)
Distances
       Distance: 0   Distance: 1   Distance: 2   Distance: 3   Distance: 4   Distance: 5
count      32581.0  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000
mean           0.0      0.074130      0.088639      0.098019      0.086721      0.090165
std            0.0      0.060757      0.061892      0.064104      0.063052      0.063065
min            0.0      0.000000      0.000000      0.000000      0.000000      0.000000
25%            0.0      0.036163      0.050704      0.058475      0.048211      0.051072
50%            0.0      0.060433      0.074099      0.082033      0.070367      0.073692
75%            0.0      0.094511      0.107353      0.117319      0.105896      0.109478
max            0.0      1.000000      1.000000      1.000000      1.000000      1.000000
Indices
         Indices: 0    Indices: 1    Indices: 2    Indices: 3    Indices: 4    Indices: 5
count  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000
mean       0.499915      0.496047      0.491692      0.490091      0.487982      0.487911
std        0.288660      0.285050      0.284879      0.285108      0.285245      0.284481
min        0.000000      0.000000      0.000000      0.000000      0.000000      0.000000
25%        0.249939      0.252072      0.244759      0.244796      0.241014      0.242679
50%        0.499693      0.492234      0.483287      0.479831      0.475429      0.472006
75%        0.749908      0.741789      0.737561      0.737306      0.734614      0.734207
max        1.000000      1.000000      1.000000      1.000000      1.000000      1.000000
Key observations
Distance: 0 is the self-distance baseline (always 0 after scaling).

Distance growth across neighbor ranks forms a stable geometric signal.

Tight neighbor bands indicate local homogeneity.

Wider bands indicate spread in local neighborhoods.

Zero distances at higher ranks indicate duplicates or near-duplicates after preprocessing.

Normalized index distributions show uniform ordering (medians near 0.5).

Manhattan distance (p=1) shifts magnitude, not neighbor ordering.

Implementation notes
Preprocessing

Numeric: SimpleImputer(strategy='mean', add_indicator=True)

Categorical: SimpleImputer(strategy='constant', fill_value='missing')

Encoding: OneHotEncoder(handle_unknown='ignore', sparse_output=False)

Scaling: MinMaxScaler

NearestNeighbors

kneighbors() returns distance and index arrays

Plots

Distance pairs: distance vs distance

Index pairs: indices vs indices

Normalization

MinMaxScaler applied before summary statistics

This project is exploratory and focuses on nearest-neighbor geometry, not production modeling.
