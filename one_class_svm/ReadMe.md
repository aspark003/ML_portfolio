# One-Class SVM

This project explores **unsupervised anomaly detection** using **One-Class Support Vector Machine (One-Class SVM)** on a mixed-type credit dataset.

The focus is on:

- Learning a boundary around normal observations  
- Inspecting decision-function behavior  
- Understanding raw score structure  
- Analyzing separation between dense core and anomaly tail  

---

## Overview

One-Class SVM is applied after a full preprocessing pipeline (numeric + categorical).

The analysis emphasizes:

- Boundary-based anomaly detection  
- Distance from learned boundary  
- Score distribution structure  
- Separation between inliers (`+1`) and outliers (`-1`)  

### Key Ideas

- The model learns a decision boundary around the majority of data  
- `predict()` returns:
  - `+1` → inlier (normal)
  - `-1` → outlier (anomaly)
- `decision_function()`:
  - Positive → inside boundary  
  - Negative → outside boundary  
- `score_samples()` returns the raw scoring values before threshold shift  

---

# Pipeline

## Data Preprocessing

### Numeric Features

- Mean imputation (+ missing indicator)  
- Min–Max scaling  

### Categorical Features

- Constant-value imputation (`"missing"`)  
- One-hot encoding (`handle_unknown='ignore'`)  

Implemented using:

- `ColumnTransformer`
- `Pipeline`

for reproducibility.

---

## Anomaly Detection (One-Class SVM)

### Configuration

```python
OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
Key Steps
fit_predict on full transformed feature matrix

Extraction of:

decision_function() scores

score_samples() raw values

Diagnostics & Analysis
Decision Function Distribution
d_function = svm.decision_function(x)
sns.kdeplot(d_function[label])
Summary:

count    32581.000000
mean        23.873220
std         14.523661
min       -108.247376
25%         15.148799
50%         25.540400
75%         34.392688
max         55.247296
Raw Score Distribution (score_samples)
s_sample = svm.score_samples(x)
sns.kdeplot(s_sample)
Summary:

count    32581.000000
mean       237.018225
std         14.523661
min        104.897630
25%        228.293804
50%        238.685406
75%        247.537694
max        268.392301
Results
Dataset sample size: 32,581 rows

Observed structure:

Strong density core

Distinct lower tail in decision function

Raw scores represent shifted boundary distance

Score spread increases with full dataset size

Dependencies
pandas

numpy

scikit-learn

matplotlib

seaborn

Scope
The goal is to:

Understand One-Class SVM boundary mechanics

Interpret decision score behavior

Visualize score structure and separation

Build intuition around margin-based anomaly detection

This is an exploration of One-Class SVM mechanics.
