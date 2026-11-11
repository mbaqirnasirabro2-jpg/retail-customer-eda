# 🛒 Retail Customer Behavior & Sales Exploratory Data Analysis (EDA)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualizations-green.svg)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Exploratory Data Analysis (EDA) project analyzing retail customer demographics, purchasing behavior, and product category performance. This project simulates a real-world data science workflow—starting from handling messy data with missing values to feature imputation, statistical aggregation, and insightful visual reporting.

---

## 📊 Project Overview & Objectives
In retail analytics, understanding customer spending patterns is vital for marketing strategies and revenue growth. This project addresses the following key business questions:
1. What is the demographic distribution of our customer base?
2. How do we systematically handle missing values and data inconsistencies in raw datasets?
3. Which product categories drive the highest average purchase volumes?
4. What correlations exist between customer income, spending score, and purchase amounts?

---

## Methodology & Pipeline Workflow

Key Data Insights & Findings
Data Quality Handling: Successfully identified and imputed missing values across Age, AnnualIncome, and ProductCategory using robust statistical methods (Median and Mode) to prevent data distortion.

Spending Behavior: Analysis reveals that higher spending scores align closely with specific demographic segments, independent of a linear age correlation.

Category Performance: High-ticket product categories like Electronics exhibit higher average purchase values per transaction compared to standard apparel goods.

## Detailed Technical Breakdown of the EDA Pipeline1.
##
Data Simulation & Missing Value Engineering (The Foundation)Real-World Simulation: In actual data science environments, data is rarely clean. Datasets often arrive with missing values (NaN), formatting errors, or noise.The Mechanism: Our script programmatically generates $500$ customer profiles with variables like Age, Annual Income, Spending Score, and Purchase Amount. To mimic real-world challenges, we intentionally injected missing values using NumPy (np.nan) across demographic and categorical features.

##2. Data Cleaning & Imputation StrategyHandling Numerical Missingness (Age, AnnualIncome):
##
Instead of deleting rows with missing values (which leads to loss of valuable data), we utilized Median Imputation. The median is robust against statistical outliers compared to the mean, ensuring income or age distributions remain unbiased.
Handling Categorical Missingness (ProductCategory): For textual categories, we applied Mode Imputation, replacing missing entries with the most frequently occurring product category. This preserves the natural frequency and categorical balance of the dataset.

##3. Statistical Aggregation & Groupby Analysis:
##
*Granular Insights:* By leveraging Pandas groupby and aggregation functions (count, mean, sum), the pipeline evaluates commercial performance across dimensions.

*Business Value:* It computes total revenue and average basket sizes per category, helping business stakeholders identify which product lines (e.g., Electronics vs. Clothing) generate high-margin transactions versus high-volume sales.

##4. Correlation & Multivariate Visualization
##
*Correlation Matrix (Heatmap):* Analyzes the linear relationship between customer income, spending score, and purchase amounts. This helps marketers determine whether wealthy customers automatically spend more or if spending behavior is driven by psychological scores.

*Distribution Histograms & Bar Plots:* Visualizes demographic spreads (such as customer age brackets) using Seaborn and Matplotlib to ensure data distributions are stable and ready for potential future machine learning classification tasks.
