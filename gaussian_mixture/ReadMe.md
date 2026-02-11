# Gaussian Mixture

This project explores **unsupervised density modeling and soft clustering** using **Gaussian Mixture Models (GMM)** on a mixed-type credit dataset.

The focus is on:

- Understanding probabilistic clustering behavior  
- Interpreting log-density structure  
- Analyzing score continuity and uniqueness  
- Inspecting soft assignment confidence  

---

## Overview

Gaussian Mixture is applied after a full preprocessing pipeline (numeric + categorical).

The analysis emphasizes:

- Modeling data as a weighted sum of Gaussian distributions  
- Log-density evaluation per sample (`score_samples`)  
- Density gradient from weak to strong regions  
- Soft cluster assignments (`predict_proba`)  
- Frequency and uniqueness of log-density values  

### Key Ideas

- Each component is a Gaussian distribution  
- Each sample receives:
  - A log-density score  
  - A probability distribution over components  
- Higher log-density - stronger density region  
- Lower log-density - weaker density region  
- Density is interpreted as a continuous spectrum  

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

for reproducibility and structured feature handling.

---

## Gaussian Mixture Configuration

```python
GaussianMixture(n_components=10,covariance_type='full',random_state=42)
Log Density Analysis
score_samples(x)

Log Density Summary
       score sample index  score sample counts
count         6502.000000          6502.000000
mean            39.125364             1.002153
std              4.029843             0.049563
min             14.541979             1.000000
25%             36.639760             1.000000
50%             37.994447             1.000000
75%             42.676317             1.000000
max             47.315675             3.000000
Interpretation
Left tail (~14–30) → weakest density

Central mass (~36–43) → dominant density region

Very high uniqueness → expected in continuous models

Maximum identical frequency = 3 (rare duplication)

The density spectrum shows smooth probabilistic structure.

KDE Visualization
Kernel Density Estimate of log-density values:

Reveals multimodal structure

Shows density peaks corresponding to dominant regions

Demonstrates separation between weaker and stronger density areas

Higher regions correspond to dense cluster cores.

Soft Assignment (predict_proba)
Each sample receives probability distribution across components.

Probability Summary
       predict proba index  predict proba count
count          2340.000000          2340.000000
mean              0.498291            27.846154
std               0.416232          1193.088792
min               0.000000             1.000000
25%               0.043165             1.000000
50%               0.495607             1.000000
75%               0.955120             1.000000
max               1.000000         57469.000000
Interpretation
Many probabilities collapse near 0 or 1

Indicates strong separation between components

Model behaves close to hard clustering in practice

Soft clustering remains available when overlap exists

Dependencies
pandas

numpy

scikit-learn

matplotlib

seaborn

Scope
The goal is to:

Understand Gaussian Mixture mechanics

Interpret log-density structure

Analyze soft clustering behavior

Build probabilistic intuition

This is an exploratory density-modeling study focused on understanding Gaussian Mixture behavior.
