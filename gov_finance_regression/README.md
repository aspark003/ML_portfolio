Residual-Based / Risk Screening Overview

This project implements an end-to-end regression-driven deviation analysis pipeline for credit score assessment.
Rather than treating regression as a prediction task, the system builds an expectation model, measures record-level deviations, assigns residuals, and risk level.

The goal is not forecasting accuracy, but quantifying material deviation and supporting audit, compliance, and risk prioritization workflows.

The output is a record-level enriched dataset designed for direct use in Power BI dashboards and human review processes.

Pipeline Architecture

The workflow is modular and executed in clearly separated stages:

Raw File → Load → Clean → Model → Residuals → Risk levely → Investigation → Dashboard

File Loading (FileLoader)

Supported formats:
csv.

Purpose:
Ensure reliable ingestion regardless of source format while preserving raw structure.

Data Cleaning & Preparation (CleanFile)

Key steps:

Applies custom header offsets where required

Normalizes column names

Removes trailing non-data rows

Adds a stable record ID

Preserves an untouched original snapshot for reconciliation

Uses ColumnTransformer for:

Numeric median imputation

Categorical missing-value handling

Feature-safe preprocessing

Expectation Modeling (LinearModel)

Regression is used only to establish an expected value, not to classify records.

Models evaluated:

Linear Regression:

linear regression r2 score:  0.78305342126522
linear regression mean absolute error:  15479.424497996435
linear regression mean squared error:  403694857.37694955

linear regression cross val score:  0.6966046668971128

Residual Calculation
Residuals are computed per record as:

residual = actual − predicted
  linear actual  linear predict  linear residual residual
0    84301.33126    50713.716274     33587.614986    under
1    39813.06005    41056.266070     -1243.206020     over
2    47298.73111    12817.304918     34481.426192    under
3   108707.77840    66951.581229     41756.197171    under
4    44454.11240    46425.983547     -1971.871147     over




random forest r2 score:  0.9867661430161156
random forest mean absolute error:  3593.6626183653348
random forest mean squared error:  24625601.55966945

random forest cross val score:  0.852320239585098

Risk Level
Risk are computed per record as:
   random forest actual  random forest prediction  random forest residual risk score
0           84301.33126              83354.148892              947.182368      under
1           39813.06005              41773.290960            -1960.230910       over
2           47298.73111              49811.221379            -2512.490269       over
3          108707.77840              96396.159127            12311.619273      under
4           44454.11240              43761.631072              692.481328      under

Evaluation approach:

Train/Test split

Cross-validation for realism checks

Full-dataset fit for expectation generation

Design principle:
Model evaluation validates plausibility — it does not drive downstream decisions.

Leakage Control

To prevent artificial performance inflation, outcome-derived or post-event fields are excluded from modeling, including:


Purpose:

Ensure real-world validity and audit defensibility.

Investigation Logic

Residual level is collapsed into a binary review gate:

Investigation required: yes

Regression ends here — all further work is analysis, not modeling.

Output for Visualization

Final output is written back to practice_original.csv and includes:

Original financial context

Model predictions

Residual values

Risk level

The dataset is Power BI–ready with no downstream model logic.

Power BI Dashboard

The dashboard supports:

Record-level inspection

Residual severity filtering

Accounting-formatted financial fields

Deviation-driven prioritization

Audit-ready interpretation

No ML logic is duplicated in Power BI — all intelligence is upstream.

Practical decision support

Residual, Risk level-based deviation analysis provides a transparent, explainable, and defensible signal suitable for financial review, audit, and compliance contexts.
