## PCA / Variance Analysis + Inverse Reconstruction

This project explores **unsupervised dimensionality analysis** using **PCA** on a CSV dataset.

The goal is to understand how PCA **redistributes variance**, where information drops off across components, and how **inverse PCA** reconstructs original features after dimensionality reduction.

---

## Pipeline

- **Numeric + Categorical preprocessing**
  - Mean imputation (+ missing indicator)
  - Min–Max scaling
  - One-hot encoding for categorical features

- **Diagnostics**
  - Explained variance ratio per component
  - Cumulative explained variance
  - Component (loading) pair comparisons
  - Inverse PCA reconstruction vs original feature space

---

## Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  

---

## Configuration used

```python
PCA(n_components=0.9,svd_solver='auto')

This configuration retains the minimum number of components required to explain ~90% of total variance.

Key Visuals
Explained variance ratio vs cumulative variance

Shows rapid variance capture in early components

Later components contribute progressively less information

PCA component (loading) pair comparisons

Visual comparison of feature contributions across principal components

Highlights which features dominate specific variance directions

Inverse PCA vs original feature pairs

Feature-space comparison of original data vs PCA-reconstructed data

Demonstrates how retained variance constrains reconstructed values

Data Summary (Derived / Scaled)
Explained variance diagnostics
       variance ratio  cumulative
count        6.000000    6.000000
mean         0.150150    0.586024
std          0.040868    0.272033
min          0.085058    0.190900
25%          0.127257    0.412037
50%          0.165000    0.619426
75%          0.176450    0.786795
max          0.190900    0.900901
Interpretation:

Roughly 6 components are sufficient to reach ~90% explained variance

Variance contribution decays steadily after the first few components

PCA scores (x_pca) summary
PC means ≈ 0 across all components
Standard deviation decreases with component index
Later PCs have tighter distributions
Interpretation:

PCA centering worked correctly

Earlier PCs carry most of the spread (variance)

Later PCs represent finer-grained structure

PCA component loadings summary
Component loadings show:
- Strong dominance of a subset of features in early components
- Later components capture weaker, more localized variance
Interpretation:

PCA axes are not uniformly influenced by all features

A small number of encoded / numeric features dominate major variance directions

Key Observations
PCA uses variance maximization during fit(); reported variance ratios are diagnostic outputs.

Earlier components preserve the majority of global structure.

Later components contribute diminishing variance and finer detail.

Inverse PCA reconstructs data in the original feature space, but:

only using variance retained by selected components

discarded variance cannot be recovered

Reconstruction plots show progressive tightening of values around dominant patterns.

PCA is lossy by design; inverse reconstruction provides a controlled way to visualize what information is removed.

Scope
This project focuses on understanding PCA behavior and diagnostics, including variance redistribution and reconstruction effects.
