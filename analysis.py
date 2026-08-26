import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Setup DuckDB connection
con = duckdb.connect(database=':memory:')

# ---------------------------------------------------------
# 1. RFM Analysis
# ---------------------------------------------------------
print("--- 1. RFM Analysis ---")
rfm_query = """
WITH clean_retail AS (
    SELECT CustomerID, InvoiceNo, CAST(InvoiceDate AS DATE) as InvoiceDate, Quantity, UnitPrice, Quantity * UnitPrice AS TotalPrice
    FROM 'data/online_retail.csv'
    WHERE CustomerID IS NOT NULL AND Quantity > 0 AND UnitPrice > 0
),
rfm_base AS (
    SELECT CustomerID, MAX(InvoiceDate) as last_purchase_date, COUNT(DISTINCT InvoiceNo) as frequency, SUM(TotalPrice) as monetary
    FROM clean_retail
    GROUP BY CustomerID
),
max_date_cte AS (
    SELECT MAX(last_purchase_date) as max_date FROM rfm_base
)
SELECT 
    b.CustomerID, CAST(m.max_date - b.last_purchase_date AS INTEGER) as recency, b.frequency, b.monetary,
    NTILE(4) OVER (ORDER BY CAST(m.max_date - b.last_purchase_date AS INTEGER) DESC) AS R_Score,
    NTILE(4) OVER (ORDER BY b.frequency ASC) AS F_Score,
    NTILE(4) OVER (ORDER BY b.monetary ASC) AS M_Score
FROM rfm_base b CROSS JOIN max_date_cte m;
"""
rfm_df = con.execute(rfm_query).df()
rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
print(rfm_df.head(3))

# ---------------------------------------------------------
# 2. Cohort Retention Analysis
# ---------------------------------------------------------
print("\n--- 2. Cohort Retention Analysis ---")
cohort_query = """
WITH clean_retail AS (
    SELECT CustomerID, CAST(InvoiceDate AS DATE) as InvoiceDate
    FROM 'data/online_retail.csv' WHERE CustomerID IS NOT NULL
),
first_purchases AS (
    SELECT CustomerID, DATE_TRUNC('month', MIN(InvoiceDate)) as cohort_month
    FROM clean_retail GROUP BY CustomerID
),
activity AS (
    SELECT 
        c.CustomerID, f.cohort_month, DATE_TRUNC('month', c.InvoiceDate) as activity_month,
        ((EXTRACT(YEAR FROM c.InvoiceDate) - EXTRACT(YEAR FROM f.cohort_month)) * 12) + (EXTRACT(MONTH FROM c.InvoiceDate) - EXTRACT(MONTH FROM f.cohort_month)) as cohort_index
    FROM clean_retail c JOIN first_purchases f ON c.CustomerID = f.CustomerID
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT CustomerID) as cohort_size FROM first_purchases GROUP BY cohort_month
),
retention AS (
    SELECT a.cohort_month, a.cohort_index, COUNT(DISTINCT a.CustomerID) as active_customers, MAX(cs.cohort_size) as cohort_size
    FROM activity a JOIN cohort_sizes cs ON a.cohort_month = cs.cohort_month
    GROUP BY a.cohort_month, a.cohort_index
)
SELECT 
    CAST(cohort_month AS DATE) as cohort_month, cohort_index, active_customers, cohort_size,
    CAST(active_customers AS FLOAT) / cohort_size as retention_rate
FROM retention ORDER BY cohort_month, cohort_index;
"""
cohort_df = con.execute(cohort_query).df()
cohort_pivot = cohort_df.pivot(index='cohort_month', columns='cohort_index', values='retention_rate')
cohort_pivot.index = pd.to_datetime(cohort_pivot.index).strftime('%Y-%m')

plt.figure(figsize=(14, 8))
sns.heatmap(cohort_pivot, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=0.5)
plt.title('Cohort Retention Heatmap')
plt.ylabel('Cohort Month')
plt.xlabel('Months Since First Purchase')
plt.tight_layout()
plt.savefig('/home/zircon/.gemini/antigravity-ide/brain/0dc9c1b0-d62c-4c82-bd98-8ed81f3f2885/cohort_heatmap.png')
print("Saved cohort heatmap.")

# ---------------------------------------------------------
# 3. A/B Testing
# ---------------------------------------------------------
print("\n--- 3. A/B Testing ---")
ab_query = """
WITH users AS (
    SELECT DISTINCT customer_id, experiment_group
    FROM 'data/events.csv'
    WHERE experiment_group IN ('Control', 'Variant_A')
),
revenue AS (
    SELECT customer_id, SUM(gross_revenue) as total_revenue
    FROM 'data/transactions.csv'
    GROUP BY customer_id
)
SELECT 
    u.customer_id, u.experiment_group, COALESCE(r.total_revenue, 0) as revenue,
    CASE WHEN r.total_revenue IS NOT NULL THEN 1 ELSE 0 END as converted
FROM users u LEFT JOIN revenue r ON u.customer_id = r.customer_id;
"""
ab_df = con.execute(ab_query).df()

control = ab_df[ab_df['experiment_group'] == 'Control']
treatment = ab_df[ab_df['experiment_group'] == 'Variant_A']

print(f"Control Size: {len(control)}")
print(f"Treatment Size: {len(treatment)}")

# Conversion Rate (Chi-Square)
conv_control = control['converted'].sum()
conv_treatment = treatment['converted'].sum()
print(f"Control Conversion Rate: {conv_control/len(control):.4f}")
print(f"Treatment Conversion Rate: {conv_treatment/len(treatment):.4f}")

chi2, p_val_chi2, _, _ = stats.chi2_contingency([
    [len(control) - conv_control, conv_control],
    [len(treatment) - conv_treatment, conv_treatment]
])
print(f"Chi2 p-value (Conversion): {p_val_chi2:.5f}")

# Revenue (T-Test)
mean_rev_control = control['revenue'].mean()
mean_rev_treatment = treatment['revenue'].mean()
print(f"Control Mean Revenue: ${mean_rev_control:.2f}")
print(f"Treatment Mean Revenue: ${mean_rev_treatment:.2f}")

t_stat, p_val_t = stats.ttest_ind(control['revenue'], treatment['revenue'], equal_var=False)
print(f"T-test Statistic: {t_stat:.4f}")
print(f"T-test p-value (Revenue): {p_val_t:.5f}")

# Effect Size (Cohen's d for revenue)
pooled_std = np.sqrt((control['revenue'].std()**2 + treatment['revenue'].std()**2)/2)
cohens_d = (mean_rev_treatment - mean_rev_control) / pooled_std
print(f"Cohen's d (Effect Size for Revenue): {cohens_d:.4f}")

# ---------------------------------------------------------
# 4. K-Means Clustering on RFM
# ---------------------------------------------------------
print("\n--- 4. K-Means Clustering ---")
rfm_features = rfm_df[['recency', 'frequency', 'monetary']].copy()
# Log transform to handle skewness
rfm_features = np.log1p(rfm_features)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)

print("Mean values by KMeans Cluster:")
print(rfm_df.groupby('KMeans_Cluster')[['recency', 'frequency', 'monetary']].mean())
