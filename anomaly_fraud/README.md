
Anomaly Detection

Unsupervised M

Project Overview

This project implements an end-to-end anomaly dashboard.

The system is designed to surface unusual patterns and material deviations, assign severity, and prioritize records for audit and investigative review using an interactive Power BI dashboard.

The goal is not fraud prediction, but risk discovery, validation, and prioritization.

Modeling Philosophy

The system separates detection from decision-making:

Machine learning identifies what is abnormal

Severity quantifies how significant the deviation is

Binary review flags drive human investigation

This design mirrors real-world audit and compliance workflows.

Modeling Components

Unsupervised Anomaly Detection (Pattern-Based)

To identify structural anomalies, the system uses model consensus across multiple density-based algorithms:

Algorithms implemented:

DBSCAN – density-based outlier detection

HDBSCAN – hierarchical density-based clustering

PCA – dimensionality reduction for clustering stability

(Isolation Forest included as a reference model)

Each model produces:

anomaly identifiers

Consensus-Based Anomaly Severity

Model agreement is combined into a single score:

Anomaly Agreement Strength

DBSCAN + OPTICS + HDBSCAN


Severity Levels

critical / low

This answers:

How confident are we that this record is structurally unusual?

Preprocessing

Median imputation (numeric)

Constant imputation (categorical)

Min-Max scaling

One-Hot Encoding

Managed via ColumnTransformer

Modeling

Independent pipelines for DBSCAN, HDBSCAN, ISOLATION FOREST

PCA applied per model (configurable components)

Cluster evaluation using:

Silhouette Score

Calinski-Harabasz Index

Davies-Bouldin Index

Dashboard Features (Power BI)

Interactive anomaly & deviation table

Row-level highlighting for:

anomaly severity

regression deviation severity

investigation status

Powerbi cards for:

id/anomaly/risk/detector

Records requiring investigation

Drill-down by:

fund

organization

project

account code

Designed to support audit review, compliance checks, and prioritization.

Output Artifacts

original.csv – enriched dataset with all anomaly and regression flags

Model evaluation metrics logged per algorithm

Power BI dashboard connected to finalized dataset

Example Model Metrics
Model	Silhouette	Calinski-Harabasz	Davies-Bouldin
DBSCAN:
silhouette_score:0.7112018964744454
calinski_harabasz_score:427.12848402501555
davies_bouldin_score:0.778474433752063

HDBSCAN 
silhouette_score:0.6154682023131522
calinski_harabasz_score:142.16463458230797
davies_bouldin_score:1.152683627699001

VARIANCE
[0.2948866  0.23071016 0.22773586 0.10002866 0.06515764]
CUMSUM
[0.2948866  0.52559676 0.75333262 0.85336128 0.91851892]

Key Takeaways

Unsupervised models identify risk signals, not conclusions

Model consensus reduces false positives

Dashboards convert ML outputs into actionable review workflows

Use Cases

Anomaly identifier

Risk prioritization

Tech Stack

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

Power BI - personal

GitHub for version control
