import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set styling for plots
sns.set_theme(style="whitegrid")
print("==================================================")
print("   RETAIL CUSTOMER DATA ANALYSIS & EDA PIPELINE   ")
print("==================================================\n")

# --- STEP 1: GENERATE SYNTHETIC DATA WITH MISSING VALUES ---
print("[INFO] Generating synthetic dataset with intentional missing values...")
np.random.seed(42)
n_rows = 500

data = {
    "CustomerID": [f"CUST_{i:03d}" for i in range(1, n_rows + 1)],
    "Age": np.random.randint(18, 70, size=n_rows),
    "AnnualIncome_k$": np.random.randint(20, 150, size=n_rows),
    "SpendingScore": np.random.randint(1, 100, size=n_rows),
    "PurchaseAmount_$": np.random.uniform(10.0, 1000.0, size=n_rows),
    "ProductCategory": np.random.choice(
        ["Electronics", "Clothing", "Home & Kitchen", "Sports"], size=n_rows
    ),
    "LoyaltyMember": np.random.choice([0, 1], size=n_rows, p=[0.6, 0.4]),
}

df = pd.DataFrame(data)

# Inject Missing Values (NaN) randomly into columns to simulate real-world data issues
missing_indices_age = np.random.choice(n_rows, size=35, replace=False)
missing_indices_income = np.random.choice(n_rows, size=50, replace=False)
missing_indices_cat = np.random.choice(n_rows, size=20, replace=False)

df.loc[missing_indices_age, "Age"] = np.nan
df.loc[missing_indices_income, "AnnualIncome_k$"] = np.nan
df.loc[missing_indices_cat, "ProductCategory"] = np.nan

# Save raw dirty data
df.to_csv("raw_retail_data_with_missing_values.csv", index=False)
print(
    f"[SUCCESS] Raw dataset created with {df.isnull().sum().sum()} total missing values!\n"
)

# --- STEP 2: DATA CLEANING & MISSING VALUE IMPUTATION ---
print("[INFO] Starting Data Cleaning and Imputation...")

# Check missing values before cleaning
print("Missing values before cleaning:\n", df.isnull().sum())

# Impute numerical columns with Median
df["Age"] = df["Age"].fillna(df["Age"].median())
df["AnnualIncome_k$"] = df["AnnualIncome_k$"].fillna(
    df["AnnualIncome_k$"].median()
)

# Impute categorical column with Mode (most frequent category)
df["ProductCategory"] = df["ProductCategory"].fillna(
    df["ProductCategory"].mode()[0]
)

print(
    f"\n[SUCCESS] Missing values after cleaning: {df.isnull().sum().sum()}\n"
)

# --- STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & STATISTICS ---
print("[INFO] Performing Statistical EDA...")

summary_stats = df.describe()
print("\n--- Summary Statistics ---")
print(summary_stats)

# Groupby analysis: Average purchase amount by product category
category_sales = (
    df.groupby("ProductCategory")["PurchaseAmount_$"]
    .agg(["count", "mean", "sum"])
    .reset_index()
)
category_sales.columns = ["ProductCategory", "Total_Orders", "Avg_Purchase_$","Total_Revenue_$"]
print("\n--- Sales Performance by Product Category ---")
print(category_sales)

# --- STEP 4: DATA VISUALIZATION ---
print("\n[INFO] Generating and saving analytical visualizations...")

# 1. Distribution of Customer Age
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True, color="blue")
plt.title("Customer Age Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution.png", dpi=300)
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
plt.savefig("correlation_heatmap.png", dpi=300)
plt.close()

# 3. Purchase Amount by Category Barplot
plt.figure(figsize=(8, 5))
sns.barplot(
    x="ProductCategory", y="PurchaseAmount_$", data=df, estimator=np.mean, palette="viridis", ci=None
)
plt.title("Average Purchase Amount per Product Category", fontsize=12, fontweight="bold")
plt.xlabel("Product Category")
plt.ylabel("Average Purchase ($)")
plt.savefig("category_purchase_barplot.png", dpi=300)
plt.close()

print("==================================================")
print(" [SUCCESS] EDA Pipeline Complete! All charts saved.")
print("==================================================")
