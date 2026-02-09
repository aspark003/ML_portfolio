## OPTICS / Ordering + Reachability

This project explores **unsupervised structure discovery** using **OPTICS** on a CSV dataset.

The goal is to extract **density structure signals** using:
- OPTICS **ordering**
- OPTICS **reachability** (as a density proxy)

---

## Pipeline

- **Numeric + Categorical preprocessing**
  - SimpleImputer (mean + indicator) + MinMaxScaler (numeric)
  - SimpleImputer (constant) + OneHotEncoder (categorical)
- **OPTICS fit**
  - `fit()` on transformed matrix
- **Reachability + ordering extraction**
  - `reachability_`
  - `ordering_`
- **Diagnostics**
  - Ordering / reachability plot
  - Normalized statistical summaries

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
Ordering / reachability scatter + line plot

Normalized reachability + ordering summary table

Data summary (Reachability / Ordering)
       reachability      ordering
count  32581.000000  32581.000000
mean       0.029096      0.500000
std        0.025821      0.288688
min        0.000000      0.000000
25%        0.017212      0.250000
50%        0.023200      0.500000
75%        0.034503      0.750000
max        1.000000      1.000000
Key observations
Reachability values stay low for most points, with spikes indicating transitions into sparser regions.

Ordering is normalized and reflects the traversal sequence chosen by OPTICS.

The reachability plot is used as a diagnostic signal, not a final clustering output.

Implementation notes
Uses OPTICS.reachability_ and OPTICS.ordering_ for diagnostics.

Reachability is normalized with MinMaxScaler before summary stats.

Ordering is normalized with MinMaxScaler before summary stats.

The plot overlays:

scatter points (ordering)

dashed line with markers (reachability)

This project focuses on interpreting OPTICS reachability structure, not building a production clustering model.
