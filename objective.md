# Project 7: Customer Segmentation (RFM) + Cohort Retention + A/B Testing

## Core Objective
Prove SQL + statistical rigor. No fancy model needed — a clean, defensible analysis beats a complex model here.

## Project Rules & Anti-Drift
- **NO ML Models for Prediction:** Do not add predictive models to "improve" this project (that is for projects 3 and 6).
- **Value Proposition:** This project's entire value is proving you can reason statistically and write clean SQL.

## Minimum Viable Product (MVP)
1. **RFM Analysis:** Computed via SQL (Recency, Frequency, Monetary via actual SQL, not just pandas `.groupby`).
2. **Cohort Retention:** Build a cohort retention table (signup month × retention%) via SQL and visualize it as a heatmap.
3. **A/B Testing:**
    - Stated hypothesis
    - Control/treatment split
    - Significance test (t-test or chi-square)
    - Effect size
    - Explicit discussion of caveats (sample size/power, novelty effect, multiple comparisons)

## Add-on
- **K-Means Clustering:** Apply K-means clustering on RFM features and compare it against the rule-based segments to see if the two methods agree.

## Proof-of-Work (Deliverables)
- A folder containing well-commented SQL scripts.
- Cohort heatmap image.
- A/B test report: Formatted as `hypothesis → test stat → p-value → CI → decision → caveats`, written as if presenting to a non-technical stakeholder.

---

## Dataset Ideation

To fulfill the requirements of this project, we need transactional data for the RFM & Cohort analysis, and experimental data for the A/B testing. We can approach this in two ways:

### Option 1: The Unified Dataset (Real Transactional + Simulated A/B Test)
Use a real-world e-commerce dataset for RFM and Cohorts, and simulate an A/B test on top of it.
- **Dataset:** [Online Retail Data Set (UCI/Kaggle)](https://archive.ics.uci.edu/ml/datasets/online+retail) or [Olist Brazilian E-Commerce (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **Why:** These datasets contain all necessary fields for RFM and Cohorts (InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID).
- **A/B Test Simulation:** We can simulate a marketing campaign (e.g., assigning 50% of our extracted RFM segments to a "Control" group and 50% to a "Treatment" group, and generating a synthetic response rate based on a known effect size). This allows us to practice the statistical rigor without needing a perfect all-in-one dataset.

### Option 2: Separate Datasets for Separate Objectives (Recommended)
Use two different datasets to cover all requirements purely with real data.
1. **For RFM & Cohorts:** [Online Retail Data Set (Kaggle)](https://www.kaggle.com/datasets/mathchi/online-retail-data-set-from-uci-ml-repo)
2. **For A/B Testing:** [Marketing A/B Testing Dataset (Kaggle)](https://www.kaggle.com/datasets/faviovazquez/marketing-ab-testing) or [Mobile Games A/B Testing (Kaggle)](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing).
- **Why:** The Marketing A/B dataset provides real control/treatment splits and conversion statuses, allowing us to perform real t-tests or chi-square tests and discuss actual sample sizes, power, and real caveats without faking the data.

**Next Steps:** Decide on Option 1 or Option 2, download the corresponding datasets, and initialize our SQL environment (e.g., SQLite, PostgreSQL, or DuckDB) to start querying.
