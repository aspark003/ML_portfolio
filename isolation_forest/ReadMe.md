# Isolation Forest 

This project explores **unsupervised anomaly detection** using **Isolation Forest** on a mixed-type credit dataset.

The focus is on:

- Extracting anomaly signals  
- Understanding decision-function behavior  
- Verifying separation between normal observations and noise  
- Analyzing score distribution structure  

---

## Overview

Isolation Forest is applied after a full preprocessing pipeline (numeric + categorical).

The analysis emphasizes:

- How random recursive partitioning isolates anomalies  
- Separation between normal points and anomalous points  
- Point-level anomaly strength via decision scores  
- Behavior under extremely small subsample sizes (`max_samples=5`)

### Key Ideas

- Anomalies require fewer splits to isolate  
- Shorter path length - Noise 
- `decision_function()` shifts scores so:
  - `score < 0` → anomaly  
  - `score ≥ 0` → normal  

Because this implementation uses **very small subsamples**, trees are extremely shallow, producing highly granular and near-unique anomaly scores.

---

# Pipeline

## Data Preprocessing

### Numeric Features

- Median imputation (+ missing indicator)  
- Min–Max scaling  

### Categorical Features

- Constant-value imputation (`"missing"`)  
- One-hot encoding (`handle_unknown='ignore'`)  

Implemented using:

- `ColumnTransformer`
- `Pipeline`

for reproducibility.

---

## Anomaly Detection (Isolation Forest)

### Configuration

```python
IsolationForest(n_estimators=200,max_samples=10,contamination=0.10,random_state=42,n_jobs=-1)
Key Steps
fit_predict on the transformed feature matrix

Extraction of:

labels_

decision_function() scores

Diagnostics & Analysis
Decision Function Separation
Scatter of sample index vs. decision score.

Highlights
Clear separation between normal (label = 1) and anomalies (label = -1)

Threshold at 0 behaves as expected

Score range reflects shallow tree depth and high randomness

Score Frequency Distribution
Unique decision values and their frequency.

Highlights
Most scores are unique

Dense central region with sparse negative tail

Negative tail corresponds to strongest anomalies

Results
Dataset Summary
Total samples: 32,581

Contamination: 10%

Labels
1 - normal

-1 - anomaly

Decision Function Summary

       sample index  decision function scores
count  32581.000000              32581.000000
mean       0.800068                  0.032961
std        0.599919                  0.024169
min       -1.000000                 -0.079209
25%        1.000000                  0.017444
50%        1.000000                  0.035272
75%        1.000000                  0.050739
max        1.000000                  0.089961
Interpretation
Majority of scores fall in the positive region

Anomalies occupy the negative tail

Numeric separation around 0 is clean and stable

Label distribution matches the 10% contamination setting

Score Distribution Summary

       unique values  count observation
count   29331.000000       29331.000000
mean        0.032437           1.110804
std         0.024358           0.388984
min        -0.079209           1.000000
25%         0.016808           1.000000
50%         0.034799           1.000000
75%         0.050394           1.000000
max         0.089961          10.000000
Interpretation
Most decision scores appear only once

Maximum frequency of 10 indicates occasional identical path lengths

Distribution is right-shifted (normal region) with a sparse negative tail

High uniqueness is expected with shallow trees (max_samples=10)

Interpretation Notes
The threshold at 0 cleanly separates normal from anomalous observations

max_samples=10 produces extremely shallow trees - high randomness and high score uniqueness

Positive region represents the dominant normal population

Negative tail captures the strongest anomalies

Score structure aligns with theoretical expectations for small-subsample Isolation Forests

Dependencies
pandas

numpy

scikit-learn

matplotlib

Scope

The goal is to:

Understand Isolation Forest mechanics

Interpret anomaly score behavior

Visualize score structure and separation

Build intuition around tree-based isolation

This is **not** a production-ready model, but an exploration of Isolation Forest mechanics.
