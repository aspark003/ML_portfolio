
## PCA – Variance Analysis & Inverse Reconstruction

This project explores **unsupervised** using **PCA** on a CSV dataset.

The goal in to explore how PCA redistributes variance, where the information drops, and how inverse PCA reconstructs featrues after dimensionality reduction.
- pipeline
  - Numeric + Categorical preprocessing (imputation, scaling, modeling)
  - PCA with n_components = 0.9, svd_solver='auto', random_state=42
  - Variance, variance ratio, cumulative variance analysis
  - Inverse PCA reconstruction for diagnostics

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib

## Configuration used
```python
PCA(
    n_components=0.9,
    svd_solver='auto',
    random_state=42,
)
```
## Key Visuals
- Variance vs variance ratio per component
- Cumulative variance vs cumulative variance ratio
- Original feature pairs vs inverse PCA feature pairs
- Progressive tightening of reconstructed features

## Data summary (derived)
```
       variance  variance ratio    cumsum  cumsum ratio
count  6.000000        6.000000  6.000000      6.000000
mean   0.614996        0.614996  0.556512      0.556512
std    0.386121        0.386121  0.383144      0.383144
min    0.000000        0.000000  0.000000      0.000000
25%    0.398703        0.398703  0.311460      0.311460
50%    0.755298        0.755298  0.603557      0.603557
75%    0.863480        0.863480  0.839287      0.839287
max    1.000000        1.000000  1.000000      1.000000
```

## Key observations
- PCA automatically uses variance during fit(); variance outputs are diagnositc only
- Inverse PCA restores featrue shape but not discarded variance
- Higher index features tighten first, revaling where information loss begins
- Inverse PCA geometry matches PCA space, not original feature spread
- PCA is lossy by design: inverse PCA reconstructs the best approximation using retained variance, making it a powerful tool for understanding where and how information is removed

This project focuses on **understanding PCA behavior**, not building a production model.
