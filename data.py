"""
data.py
-------
Standalone exploratory data analysis script demonstrating data loading,
statistical metrics, distribution plots, and hypothesis testing (t-test)
on the placement dataset.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

from ml.data_loader import load_data, clean_data

# Load dataset using data_loader
raw_df = load_data()
df = clean_data(raw_df)

print("--- Head ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Shape & Missing Values ---")
print("Shape:", df.shape)
print("Null counts:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\n--- Placement Target Counts ---")
print(df["PlacementStatus"].value_counts())

# Feature and Target Split
target = "PlacementStatus"
features = ["CGPA", "SoftSkillsRating", "CodingTestScore", "Projects"]
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain/Test Split Shapes:", X_train.shape, X_test.shape)

print("\n--- CGPA Stats ---")
print("CGPA Mean:", df["CGPA"].mean())
print("CGPA Median:", df["CGPA"].median())
print("CGPA Mode:", df["CGPA"].mode()[0])
print("Stream Mode:", df["Stream"].mode()[0])

num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns
print("\nNumeric Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())

# Categorize CGPA into bins
bins = [0, 6, 8, 10]
labels = ["Low", "Medium", "High"]
df["CGPA_Category"] = pd.cut(df["CGPA"], bins=bins, labels=labels, ordered=True)
print("\nCGPA Categories Preview:")
print(df[["CGPA", "CGPA_Category"]].head())

# CDA / Hypothesis Testing: T-Test comparing CGPA of Placed vs Unplaced students
placed_cgpa = df[df["PlacementStatus"] == 1]["CGPA"]
unplaced_cgpa = df[df["PlacementStatus"] == 0]["CGPA"]
t_stat, p_val = stats.ttest_ind(placed_cgpa, unplaced_cgpa)
print(f"\n--- T-Test (CGPA of Placed vs Unplaced) ---")
print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")

# Range, Variance, Std Dev, IQR for CGPA
cgpa_range = df["CGPA"].max() - df["CGPA"].min()
cgpa_var = df["CGPA"].var()
cgpa_std = df["CGPA"].std()
q1, q3 = df["CGPA"].quantile(0.25), df["CGPA"].quantile(0.75)
iqr = q3 - q1

print(f"\nCGPA Spread: Range={cgpa_range:.2f}, Var={cgpa_var:.4f}, Std={cgpa_std:.4f}, IQR={iqr:.4f}")

# Visualizations (save to chart for headless execution)
os.makedirs("static/charts", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

sns.histplot(df["CGPA"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("CGPA Distribution")

sns.boxplot(x=df["CGPA"], ax=axes[0, 1])
axes[0, 1].set_title("CGPA Boxplot")

sns.boxplot(x="PlacementStatus", y="CGPA", data=df, ax=axes[1, 0])
axes[1, 0].set_title("CGPA by Placement Status")

sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("Feature Correlation")

plt.tight_layout()
chart_path = "static/charts/data_script_summary.png"
plt.savefig(chart_path)
plt.close(fig)
print(f"\nSaved visualization to {chart_path}")