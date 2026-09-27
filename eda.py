import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("inventory_stockout_risk_dataset.csv")

print("========== DATASET INFORMATION ==========")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== STOCKOUT RISK ==========")
print(df["stockout_risk"].value_counts())

print("\nPercentage:")
print(df["stockout_risk"].value_counts(normalize=True) * 100)

print("\n========== CATEGORY ==========")
print(df["category"].value_counts())

print("\n========== RISK BY CATEGORY ==========")
print(pd.crosstab(df["category"], df["stockout_risk"]))

print("\n========== RISK BY SEASON ==========")
print(pd.crosstab(df["season"], df["stockout_risk"]))


# Stockout Risk Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="stockout_risk")
plt.title("Stockout Risk Distribution")
plt.xlabel("Stockout Risk")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.show()


# Current Stock Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["current_stock"], bins=30, kde=True)
plt.title("Current Stock Distribution")
plt.xlabel("Current Stock")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Daily Sales Rate
plt.figure(figsize=(8, 4))
sns.histplot(df["sales_rate_daily"], bins=30, kde=True)
plt.title("Daily Sales Rate Distribution")
plt.xlabel("Sales Rate per Day")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Lead Time
plt.figure(figsize=(8, 4))
sns.histplot(df["lead_time_days"], bins=20, kde=True)
plt.title("Supplier Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Stock vs Sales Rate
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="current_stock",
    y="sales_rate_daily",
    hue="stockout_risk"
)
plt.title("Current Stock vs Daily Sales Rate")
plt.xlabel("Current Stock")
plt.ylabel("Daily Sales Rate")
plt.tight_layout()
plt.show()


# Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
