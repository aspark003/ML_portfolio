## PCA / Variance Analysis + Inverse Reconstruction

This project explores **unsupervised dimensionality analysis** using **PCA** on a CSV dataset.

The goal is to see how PCA **redistributes variance**, where information drops off, and how **inverse PCA** reconstructs features after reduction.

---

## Pipeline

- **Numeric + Categorical preprocessing**
  - Imputation
  - Scaling
  - Encoding

- **Diagnostics**
  - Component variance + variance ratio
  - Cumulative variance + cumulative ratio
  - Inverse PCA reconstruction comparisons

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  

---

## Configuration used
```python
PCA(n_components=0.9,svd_solver='auto',random_state=42)
Key Visuals
Variance vs variance ratio per component

Cumulative variance vs cumulative variance ratio

Original feature pairs vs inverse PCA feature pairs

Progressive tightening of reconstructed features

Data summary (Derived / Scaled)
       variance  variance ratio    cumsum  cumsum ratio
count  6.000000        6.000000  6.000000      6.000000
mean   0.614996        0.614996  0.556512      0.556512
std    0.386121        0.386121  0.383144      0.383144
min    0.000000        0.000000  0.000000      0.000000
25%    0.398703        0.398703  0.311460      0.311460
50%    0.755298        0.755298  0.603557      0.603557
75%    0.863480        0.863480  0.839287      0.839287
max    1.000000        1.000000  1.000000      1.000000
Key observations
PCA uses variance during fit(); reported variance values are diagnostic outputs.

Inverse PCA restores feature shape but cannot restore discarded variance.

Earlier components preserve the largest variance; later components contribute less.

Inverse PCA reconstructs the best approximation using only retained variance.

PCA is lossy by design; inverse reconstruction is a controlled way to see where information is removed.

This project focuses on understanding PCA behavior, not building a production model.
