# Customer Segmentation and A/B Testing Analysis

## Project Overview
This repository contains a comprehensive data analytics pipeline designed to evaluate customer behavior and marketing performance. The objective is to demonstrate rigorous statistical methodology and robust SQL data manipulation to extract actionable insights from transactional and event-level data.

## Methodology

### 1. RFM Analysis & Cohort Retention
- **Data Engine:** DuckDB is utilized to process raw CSV data efficiently.
- **RFM Segmentation:** Customers are segmented into quartiles based on Recency, Frequency, and Monetary value utilizing SQL window functions (`NTILE(4)`).
- **Cohort Retention:** Retention metrics are calculated via SQL-based month-over-month cohort analysis. The output is a matrix detailing the percentage of active customers retained over time.

### 2. A/B Testing Evaluation
- **Hypothesis:** Variant A provides a statistically significant improvement in conversion rates and average revenue per user compared to the Control group.
- **Metrics Evaluated:**
  - Conversion Rate (Chi-Square Test)
  - Average Revenue per User (Two-Sample T-Test)
- **Effect Size:** Cohen's d is calculated to evaluate the practical significance of the revenue difference.

### 3. K-Means Clustering Validation
- **Clustering Approach:** Standard K-Means (k=4) is applied to the log-transformed RFM features.
- **Purpose:** To validate the manual rule-based quartiles against unsupervised machine learning segments, ensuring robust persona definition (e.g., At-Risk, Loyal, New, Champions).

## Key Findings

1. **Retention:** Initial retention drops significantly after the first month but stabilizes into a core returning user base.
2. **A/B Test Results:** The statistical analysis indicates no significant difference between the Control and Variant A.
   - Conversion Rate p-value: 0.51
   - Revenue p-value: 0.64
   - Cohen's d (Effect Size): 0.0021
   - **Conclusion:** Do not deploy Variant A. The observed variance is negligible.

## Usage Instructions

### Environment Setup
The project requires the `ml_core` conda environment containing `pandas`, `duckdb`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, and `streamlit`.

### Execution
To execute the core analytics pipeline:
```bash
conda run -n ml_core python analysis.py
```

To launch the interactive dashboard:
```bash
conda run -n ml_core streamlit run dashboard.py
```
