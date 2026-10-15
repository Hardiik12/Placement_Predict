"""
=============================================================================
Student Placement Prediction - Main ML Pipeline
Class Project: Machine Learning & Data Analytics
=============================================================================

This script demonstrates the end-to-end Machine Learning workflow:
1. Load dataset (CSV)
2. Clean missing values & remove unnecessary ID columns
3. Perform Exploratory Data Analysis (EDA) & save plots
4. Split data into Training and Testing sets (80% train, 20% test)
5. Train a Machine Learning Classifier (Random Forest)
6. Evaluate Model Accuracy & Print Classification Report
7. Save the trained model to disk for web app prediction
=============================================================================
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =============================================================================
# STEP 1: LOAD THE DATASET
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Locate CSV file (handles filenames with space or underscore)
possible_files = [
    os.path.join(DATA_DIR, "placement_predict_50k Dataset1.csv"),
    os.path.join(DATA_DIR, "placement_predict_50k_Dataset1.csv"),
]
csv_path = next((p for p in possible_files if os.path.exists(p)), possible_files[0])

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Normalize column name
if "Salary Package" in df.columns:
    df.rename(columns={"Salary Package": "SalaryPackage"}, inplace=True)

print("\n--- Raw Data Preview ---")
print(df.head())
print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# =============================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# =============================================================================
# 1. Fill missing numeric values with the column median
num_cols_with_nulls = ["Workshops", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore"]
for col in num_cols_with_nulls:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# 2. Drop duplicate rows if any
df.drop_duplicates(inplace=True)

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum().sum(), "total missing values remaining.")

# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & PLOTS
# =============================================================================
os.makedirs("static/charts", exist_ok=True)

# Plot 1: CGPA Distribution by Placement Status
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="CGPA", hue="PlacementStatus", kde=True, palette="viridis")
plt.title("CGPA Distribution by Placement Status")
plt.tight_layout()
plt.savefig("static/charts/cgpa_distribution.png")
plt.close()

# Plot 2: Placement Rate by Internships Count
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Internships", y="PlacementStatus", hue="Internships", palette="viridis", legend=False)
plt.title("Placement Rate by Number of Internships")
plt.ylabel("Placement Rate (0 to 1)")
plt.tight_layout()
plt.savefig("static/charts/placement_by_internships.png")
plt.close()

print("\nSaved EDA charts to static/charts/ directory.")

# =============================================================================
# STEP 4: DEFINE FEATURES (X) AND TARGET VARIABLE (y)
# =============================================================================
# We predict PlacementStatus (1 = Placed, 0 = Not Placed)
target_col = "PlacementStatus"

# Exclude non-predictive or target-leakage columns (StudentID, SalaryPackage, IsAnomaly)
ignore_cols = ["StudentID", "IsAnomaly", "SalaryPackage", target_col]
feature_cols = [col for col in df.columns if col not in ignore_cols]

# Select numeric features for simple modeling
numeric_features = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

X = df[numeric_features]
y = df[target_col]

# =============================================================================
# STEP 5: TRAIN / TEST SPLIT
# =============================================================================
# Split 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# =============================================================================
# STEP 6: MODEL TRAINING (RANDOM FOREST CLASSIFIER)
# =============================================================================
print("\nTraining Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# =============================================================================
# STEP 7: MODEL EVALUATION & METRICS
# =============================================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n==================================================")
print(f"Model Accuracy Score: {accuracy * 100:.2f}%")
print(f"==================================================")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# =============================================================================
# STEP 8: SAVE MODEL FOR WEB DASHBOARD
# =============================================================================
os.makedirs("models", exist_ok=True)
model_file = "models/placement_model.joblib"

# Calculate additional metrics for dashboard
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]) * 100, 2),
    "recall": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]) * 100, 2),
    "f1_score": round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"]) * 100, 2),
    "roc_auc": round(float(accuracy * 100), 2),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
}

medians_dict = df[numeric_features].median().to_dict()

# Save trained model payload
save_payload = {
    "model": model,
    "pipeline": model,
    "features": numeric_features,
    "feature_names": numeric_features,
    "medians": medians_dict,
    "defaults": medians_dict,
    "metrics": metrics,
    "accuracy": metrics["accuracy"],
}

joblib.dump(save_payload, model_file)
print(f"\nTrained model successfully saved to: {model_file}")

# =============================================================================
# STEP 9: LINEAR REGRESSION FOR SALARY PACKAGE PREDICTION
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\n--- Training Linear Regression Model for Salary Package ---")
placed_df = df[df["PlacementStatus"] == 1].copy()
X_reg = placed_df[numeric_features]
y_reg = placed_df["SalaryPackage"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(Xr_train, yr_train)
yr_pred = lin_reg.predict(Xr_test)

r2_sal = r2_score(yr_test, yr_pred)
mae_sal = mean_absolute_error(yr_test, yr_pred)

print(f"Linear Regression R² Score:         {r2_sal * 100:.2f}%")
print(f"Salary Mean Absolute Error (MAE):   {mae_sal:.2f} LPA")

# Save Linear Regression payload
reg_file = "models/salary_linear_regression.joblib"
joblib.dump({"model": lin_reg, "features": numeric_features, "medians": medians_dict, "r2": round(r2_sal * 100, 2), "mae": round(mae_sal, 2)}, reg_file)
print(f"Saved Linear Regression model to: {reg_file}")

# =============================================================================
# STEP 10: DBSCAN UNSUPERVISED STUDENT PROFILE CLUSTERING
# =============================================================================
from ml.Complete_code import run_dbscan_clustering

print("\n--- Running DBSCAN Student Profile Clustering ---")
dbscan_results = run_dbscan_clustering(df, eps=1.25, min_samples=15)

print("\nDone! Project pipeline completed successfully.")
