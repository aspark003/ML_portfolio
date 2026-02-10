## OPTICS / Ordering + Reachability

This project explores **unsupervised structure discovery** using **OPTICS** on a CSV dataset.

The goal is to extract **density structure signals** using:
- OPTICS **ordering**
- OPTICS **reachability** (used as a density proxy)

---

## Pipeline

- **Numeric + Categorical preprocessing**
  - SimpleImputer (mean + indicator) + MinMaxScaler (numeric)
  - SimpleImputer (constant) + OneHotEncoder (categorical)

- **OPTICS fit**
  - `fit()` on the transformed feature matrix

- **Reachability + ordering extraction**
  - `reachability_`
  - `ordering_`

- **Diagnostics**
  - OPTICS reachability plot (reachability ordered by traversal)
  - Normalized statistical summaries for reachability and ordering

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  

---

## Configuration used

```python
OPTICS(min_samples=5,metric='minkowski',p=2,algorithm='auto')

Key Visuals
OPTICS reachability plot

Reachability distance plotted against OPTICS ordering position

Scatter points with an overlaid dashed line

Reachability structure

Low reachability valleys indicate dense regions

Sharp spikes indicate transitions into sparse regions or noise

Data Summary (Reachability / Ordering)
       reachability      ordering
count  32581.000000  32581.000000
mean       0.029105      0.500000
std        0.026108      0.288688
min        0.000000      0.000000
25%        0.017212      0.250000
50%        0.023200      0.500000
75%        0.034503      0.750000
max        1.000000      1.000000
Reachability values are normalized using MinMaxScaler for summary statistics.

Key Observations
Reachability values remain low for most points, indicating dense local neighborhoods.

Pronounced spikes in reachability correspond to transitions into sparser regions.

Repeating valley–spike patterns indicate multiple density-separated structures.

OPTICS ordering defines the traversal sequence and is not itself a distance metric.

The reachability plot serves as a diagnostic signal, not a final clustering output.

Implementation Notes
Uses OPTICS.reachability_ and OPTICS.ordering_ as primary diagnostic outputs.

Infinite reachability values are replaced with the maximum finite reachability.

Reachability is reordered using OPTICS ordering before visualization.

Summary statistics normalize reachability for comparability.

Ordering is normalized only for tabular summaries, not for interpretation.

Scope
This project focuses on interpreting OPTICS reachability structure and density transitions, not on producing finalized cluster labels or a production clustering model.
