Local Outlier Factor (LOF)
This project explores unsupervised anomaly detection using Local Outlier Factor (LOF) on a mixed-type credit dataset.

The focus is on:

Density-based anomaly detection

Understanding local neighborhood structure

Inspecting LOF score behavior

Analyzing separation using density ratios

Overview
Local Outlier Factor is applied after a full preprocessing pipeline (numeric + categorical).

The analysis emphasizes:

Local density comparison

Ratio between neighbor density and point density

Score distribution structure

Separation between inliers and outliers using a learned threshold

Key Ideas
LOF compares each point’s local density to its neighbors

negative_outlier_factor_ provides the anomaly score

More negative values indicate stronger outliers

offset_ is used as the classification threshold

Classification rule:

Score < offset → outlier

Score ≥ offset → inlier

Pipeline
Data Preprocessing
Numeric Features
Mean imputation (+ missing indicator)

Min–Max scaling

Categorical Features
Constant-value imputation ("missing")

One-hot encoding (handle_unknown='ignore')

Implemented using:

ColumnTransformer

Pipeline

for reproducibility.

Anomaly Detection (Local Outlier Factor)
Configuration
Code
LocalOutlierFactor(
    n_neighbors=5,
    contamination=0.10,
    metric='minkowski',
    p=1,
    leaf_size=30
)
Distance Metric
p=1 → Manhattan distance

Local density estimated using k-nearest neighbors

Density ratio defines anomaly strength

Diagnostics & Analysis
LOF Score Distribution
Code
nof = lof.negative_outlier_factor_
l_x, l_y = np.unique(nof, return_counts=True)
plt.scatter(l_x, l_y)
Interpretation:

X-axis → LOF score

Y-axis → Frequency (count of identical scores)

Most scores appear once (continuous values)

Extreme negative scores represent strong outliers

Example summary:

Code
count    32018.000000
mean        -1.078619
std          0.135330
min         -3.455863
25%         -1.111745
50%         -1.042895
75%         -1.000982
max         -0.843146
Observations:

Majority of scores cluster near −1

Strong outliers extend into lower tail

Score spread reflects density variation

Classification Visualization
Code
plt.scatter(np.argsort(l_x[l_x > offset]), l_x[l_x > offset], marker='x')
plt.scatter(np.argsort(l_x[l_x < offset]), l_x[l_x < offset], marker='o')
Interpretation:

Sorted scores improve visual clarity

offset_ separates inliers from outliers

Marker separation shows density-based classification

Results
Dataset sample size: ~32,000 rows

Observed structure:

Strong dense core near −1

Long negative tail

Clear threshold-based separation

Score distribution consistent with density-based modeling

Dependencies
pandas

numpy

scikit-learn

matplotlib

seaborn

Scope
The goal is to:

Understand LOF density mechanics

Interpret negative_outlier_factor_ behavior

Visualize density-based separation

Build intuition around neighborhood-driven anomaly detection
