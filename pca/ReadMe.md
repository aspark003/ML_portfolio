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
  - PCA score distribution
  - PCA component (loading) distribution
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
PCA(n_components=0.9, svd_solver='auto')
Key Visuals
Variance ratio vs cumulative variance per component

PCA component (loading) pair comparisons

Original feature pairs vs inverse PCA feature pairs

Progressive tightening of reconstructed features

Signals (Numeric Outputs)
Variance Ratio & Cumulative Variance
     variance ratio  cumulative
count        6.000000    6.000000
mean         0.150150    0.586024
std          0.040868    0.272033
min          0.085058    0.190900
25%          0.127257    0.412037
50%          0.165000    0.619426
75%          0.176450    0.786795
max          0.190900    0.900901
Signal meaning:

~6 principal components are sufficient to reach ~90% cumulative variance

Variance contribution decreases steadily across components

No single component dominates variance completely

PCA Scores (x_pca) Distribution
               PC:1          PC:2          PC:3          PC:4          PC:5          PC:6
count  3.258100e+04  3.258100e+04  3.258100e+04  3.258100e+04  3.258100e+04  3.258100e+04
mean  -5.582976e-17  8.614357e-17 -4.754253e-17  3.031381e-17 -3.489360e-18  7.022337e-17
std    4.404648e-01  4.262152e-01  4.151135e-01  4.038020e-01  3.436396e-01  2.940125e-01
min   -4.979258e-01 -5.720738e-01 -6.499181e-01 -4.223867e-01 -2.056817e-01 -1.122257e-01
25%   -1.918996e-01 -2.838948e-01 -1.469256e-01 -2.154964e-01 -1.381566e-01 -1.042100e-01
50%   -1.424125e-01 -1.518239e-01 -7.715801e-02 -1.139342e-01 -1.126991e-01 -9.332759e-02
75%   -4.231166e-02  2.096983e-01  1.666330e-01  4.982508e-02 -8.818796e-02 -8.900248e-02
max    8.493354e-01  7.395878e-01  7.470737e-01  8.779869e-01  1.017550e+00  9.443641e-01
Signal meaning:

Means ≈ 0 confirm proper centering

Standard deviation decreases with component index

Early PCs carry broader spread; later PCs are tighter

PCA Component Loadings Distribution
       COMPONENT:1  COMPONENT:2  COMPONENT:3  COMPONENT:4  COMPONENT:5  COMPONENT:6  COMPONENT:7  COMPONENT:8  COMPONENT:9  COMPONENT:10  COMPONENT:11
count     6.000000     6.000000     6.000000     6.000000     6.000000     6.000000     6.000000     6.000000     6.000000      6.000000      6.000000
mean      0.005652     0.002915    -0.000046     0.002577     0.159588     0.091663     0.127180     0.156564    -0.027763     -0.049599     -0.298045
std       0.013968     0.006384     0.001942     0.018762     0.411608     0.395693     0.383514     0.369983     0.407061      0.404586      0.245062
min      -0.009970    -0.003924    -0.002178    -0.026157    -0.030958    -0.225912    -0.145124    -0.030111    -0.449711     -0.433194     -0.649360
25%       0.001381     0.000454    -0.001586    -0.004838    -0.010772    -0.126291    -0.092699    -0.013170    -0.160474     -0.247651     -0.465407
50%       0.003015     0.001455    -0.000317     0.002556    -0.003103    -0.046986    -0.043944     0.025144    -0.145076     -0.162544     -0.206090
75%       0.004505     0.002927     0.001649     0.011924     0.005736     0.113771     0.171504     0.032892    -0.044468     -0.029119     -0.175117
max       0.032011     0.014964     0.002240     0.028761     0.999341     0.855232     0.861725     0.909672     0.749825      0.721350     -0.010095
Signal meaning:

Early components show stronger feature dominance

Loadings include both positive and negative contributions

Later components capture weaker, more localized variance

Key Observations
PCA uses variance during fit(); variance ratios are diagnostic outputs.

Early components preserve the majority of global structure.

Later components contribute diminishing variance.

Inverse PCA reconstructs feature space using only retained variance.

Discarded variance cannot be recovered.

Reconstruction plots show tightening toward dominant variance directions.

Scope
This project focuses on understanding PCA behavior and diagnostics, not building a production model.
