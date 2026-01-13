Power BI Anomaly Detection Dashboard (Unsupervised ML)
Overview

This project demonstrates an end-to-end analytics workflow using Power BI Desktop combined with machine learning–driven anomaly detection in Python.

The dashboard analyzes structured, financial-style data and automatically flags abnormal patterns, risk levels, and outliers using unsupervised learning techniques. The goal is to simulate real-world operational and compliance scenarios where labels are unavailable, but unusual behavior must still be detected and prioritized.

Real-World Scenarios Simulated

Fraud detection

Risk scoring

Operational anomaly monitoring

Behavioral deviation analysis

Dashboard Features

Interactive anomaly detection dashboard in Power BI

Model-driven outlier detection (unsupervised)

Risk level classification (Low / Medium / High)

Drilldowns by business attributes (ex: region, vendor/account type, department)

Record-level investigation table for audit review

Machine Learning Approach

Unsupervised anomaly detection techniques used to score and flag abnormal behavior:

PCA (dimensionality reduction)

HDBSCAN (density-based clustering / anomaly labeling)

Isolation Forest (outlier detection)

Outputs include anomaly flags and severity scoring that feed directly into the Power BI dashboard.

Power BI Desktop

Python (pandas, NumPy, scikit-learn)

Machine Learning: HDBSCAN, Isolation Forest, PCA

Data preprocessing and feature engineering

Dashboard development and KPI reporting

Use Cases

Fraud & compliance monitoring

Financial risk analysis

Vendor / account anomaly detection

Operational intelligence & outlier monitoring

Audit and governance dashboards

Author

Antonio Park

Machine Learning Engineer

ML Portfolio: https://github.com/aspark003/ML_portfolio
