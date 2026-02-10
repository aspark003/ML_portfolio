## NEAREST NEIGHBORS / kneighbors

This project explores **unsupervised structure discovery** using **nearest neighbors** on a CSV dataset.

The goal is to understand **nearest-neighbor geometry and neighborhood structure** as a signal, including how local neighborhoods behave and how **structural irregularities can indicate noise or boundary points**.

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
  - Neighbor index structure scatter (point index vs k-th neighbor index)
  - Distance summaries across neighbor ranks
  - Index summaries across neighbor ranks

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  

---

## Configuration used

```python
NearestNeighbors(n_neighbors=6,metric='minkowski',p=1,algorithm='auto',leaf_size=30,n_jobs=-1)
Key Visuals
k-NN neighborhood structure

Point index vs k-th nearest neighbor index

Reveals neighborhood assignment patterns and structural discontinuities

Nearest-neighbor distance summaries

Distance growth across neighbor ranks

Identification of tight vs diffuse local neighborhoods

Nearest-neighbor index summaries

Distribution of neighbor indices across the dataset

Structural ordering and dispersion diagnostics

Data Summary (Distance / Indices)
Distances
       Distance: 1   Distance: 2   Distance: 3   Distance: 4   Distance: 5   Distance: 6
count      32581.0  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000
mean           0.0      0.052337      0.070225      0.081336      0.089621      0.096403
std            0.0      0.042895      0.049034      0.053194      0.056401      0.059002
min            0.0      0.000000      0.000000      0.000000      0.012048      0.012048
25%            0.0      0.025531      0.040171      0.048522      0.055174      0.059829
50%            0.0      0.042666      0.058705      0.068071      0.074992      0.080992
75%            0.0      0.066726      0.085051      0.097351      0.106774      0.114472
max            0.0      0.706013      0.792260      0.829800      0.906562      0.947612
Indices
         Indices: 1    Indices: 2    Indices: 3    Indices: 4    Indices: 5    Indices: 6
count  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000  32581.000000
mean   16287.221203  16162.214634  16019.833216  15970.229735  15898.953439  15896.158528
std     9404.535125   9286.355290   9281.064865   9287.119671   9292.990180   9267.809117
min        0.000000      2.000000      1.000000      6.000000      1.000000      1.000000
25%     8143.000000   8214.000000   7975.000000   7980.000000   7853.000000   7907.000000
50%    16280.000000  16038.000000  15746.000000  15636.000000  15490.000000  15378.000000
75%    24432.000000  24168.000000  24030.000000  24023.000000  23934.000000  23920.000000
max    32580.000000  32580.000000  32580.000000  32580.000000  32580.000000  32579.000000
Key Observations
Distance: 1 is the self-distance baseline (always 0 after scaling).

Distance increases monotonically with neighbor rank, forming a stable geometric signal.

Tight distance bands indicate locally homogeneous neighborhoods.

Wider distance spread indicates variation in local density.

Zero or near-zero distances at higher ranks indicate duplicates or near-duplicates after preprocessing.

Neighbor index distributions span the full dataset range, indicating no trivial ordering bias.

Structural Interpretation (k-NN Index Plot)
The k-th neighbor index scatter reveals neighborhood assignment structure rather than raw distance.

Dense rectangular regions indicate stable neighborhood blocks.

Abrupt transitions and scattered regions indicate boundary points or structurally isolated samples.

Points with irregular neighbor index behavior often correspond to noise, edge cases, or low-density regions.

This visualization highlights structural irregularities that are not visible in distance-only plots.

Implementation Notes
Preprocessing
Numeric: SimpleImputer(strategy='mean', add_indicator=True)

Categorical: SimpleImputer(strategy='constant', fill_value='missing')

Encoding: OneHotEncoder(handle_unknown='ignore', sparse_output=False)

Scaling: MinMaxScaler

NearestNeighbors
kneighbors() returns:

distance: local geometric separation

indices: neighborhood structure and assignment

Plots
Structure plot: point index vs k-th neighbor index

Summary tables: distance and index distributions

Scope
This project is exploratory and focuses on nearest-neighbor geometry and structural diagnostics, not production modeling.
