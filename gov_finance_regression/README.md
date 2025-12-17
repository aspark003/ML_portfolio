https://github.com/aspark003/ML_portfolio/

https://app.fabric.microsoft.com/groups/me/dashboards/46109454-2c02-4f57-81d8-533ddb1b16d3?experience=fabric-developer


FY26 Funds Closeout Risk Dashboard
Overview

This project builds a data-driven fiscal year (FY) closeout review system that helps identify overspending, underspending, and unusual fund usage patterns before the fiscal year closes.

It uses multiple regression models to estimate expected spending behavior and then measures how far actual spending deviates from those expectations.
The results are summarized into simple risk categories that support audit review, budget reconciliation, and closeout decisions.

The final output is visualized in a Power BI dashboard titled:

“FY 26 Close Out Dashboard”

What Problem This Solves

At the end of a fiscal year:

Unused funds can be lost

Rushed spending can increase risk

Manual review is slow and subjective

This system answers:

Where was more money used than expected?

Where was less money used than expected?

Which records need review before closeout?

How the System Works (Plain Language)
Step 1: Load and Validate Data

The FileLoader class:

Safely loads Excel, CSV, TSV, JSON, or text files

Verifies Excel file integrity

Standardizes the input into a clean DataFrame

Saves a normalized CSV for modeling

Purpose: Ensure the data is usable and consistent.

Step 2: Clean and Prepare Data

The CleanFile class:

Standardizes column names

Creates a unique id

Removes non-modeling fields

Handles missing values using median imputation

Prepares the dataset for regression modeling

Purpose: Remove noise and focus on spending behavior.

Step 3: Predict Expected Spending

Three different regression models are used to estimate how much funding should have been used:

Ridge Regression

Linear, conservative baseline

Captures normal spending trends

GridSearch-Tuned Random Forest

Nonlinear, pattern-aware

Captures structured deviations

XGBoost

Boosted, error-focused

Captures complex spending behavior

Each model answers:

“Based on similar records, how much money should have been used?”

Step 4: Measure Deviation (Residuals)

For each model:

Residual = Actual Funds Used − Predicted Funds Used


This tells us:

Positive residual → more money used than expected

Negative residual → less money used than expected

Residuals are the core signal of this system.

Step 5: Convert Residuals into Risk Labels

Residuals are converted into relative risk labels using data-driven percentiles:

High → Top 25% of deviations

Average → Middle range

Low → Bottom range

This avoids hard-coded thresholds and adapts to the data.

Step 6: Final Risk Classification

The final decision layer:

Uses the distribution of model deviations

Assigns a final risk category:

High

Average

Low

This produces a clear, audit-friendly outcome for each record.

What Was Found

The system identified:

Records where funds were used far above expectations (High)

Records where funds were partially or not fully used (Low)

Records that behaved normally (Average)

This does not label fraud.
It identifies where attention is needed during FY closeout.

Dashboard Output

The Power BI dashboard displays:

Total records reviewed

Total funds analyzed

Overall risk indicator

Detailed transaction-level table

Clear high / average / low classification

This allows:

Rapid closeout review

Targeted follow-up

Reduced manual effort

Why This Approach Is Useful

Uses multiple independent models

Avoids single-model bias

Data-driven thresholds (no guessing)

Easy to explain to non-technical users

Scales to future fiscal years

Key Takeaway

This system helps ensure fiscal year closeout is intentional, balanced, and reviewable by highlighting where funds were overused, underused, or behaved unusually.

