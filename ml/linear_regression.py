"""
=============================================================================
Student Salary Prediction - Linear Regression
Class Project: Machine Learning & Data Analytics
=============================================================================

This script demonstrates applying Linear Regression to predict the continuous
Salary Package (SalaryPackage in LPA) based on student academic & skill attributes.

Workflow:
1. Load & clean placement dataset
2. Filter placed students (SalaryPackage > 0)
3. Define features (X: CGPA, CodingTestScore, SoftSkills, Internships, etc.)
4. Define target (y: SalaryPackage)
5. Split into train (80%) and test (20%) sets
6. Train LinearRegression model
7. Compute evaluation metrics (R2 Score, MAE, MSE, RMSE)
8. Save regression model artifact to models/salary_linear_regression.joblib
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.data_loader import clean_data, load_data


def train_salary_linear_regression():
    """Train Linear Regression model to predict Salary Package in LPA."""
    raw_df = load_data()
    df = clean_data(raw_df)

    # Filter placed students with non-zero salary package
    placed_df = df[df["PlacementStatus"] == 1].copy()

    # Features and Target
    feature_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "CodingTestScore",
        "SoftSkillsRating",
        "MockInterviewScore",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "Publications",
    ]
    feature_cols = [c for c in feature_cols if c in placed_df.columns]

    X = placed_df[feature_cols]
    y = placed_df["SalaryPackage"]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("==================================================")
    print("Linear Regression Results (Salary Package Prediction)")
    print("==================================================")
    print(f"R² Score (Accuracy Variance Explained): {r2 * 100:.2f}%")
    print(f"Mean Absolute Error (MAE):            {mae:.2f} LPA")
    print(f"Root Mean Squared Error (RMSE):        {rmse:.2f} LPA")
    print("==================================================")

    print("\n--- Feature Coefficients ---")
    coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": model.coef_})
    coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="Abs_Coef", ascending=False).drop(columns=["Abs_Coef"])
    print(coef_df.to_string(index=False))

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "salary_linear_regression.joblib")
    payload = {
        "model": model,
        "features": feature_cols,
        "medians": placed_df[feature_cols].median().to_dict(),
        "intercept": float(model.intercept_),
        "metrics": {"r2": round(r2 * 100, 2), "mae": round(mae, 2), "rmse": round(rmse, 2)},
    }
    joblib.dump(payload, model_path)
    print(f"\nSaved Linear Regression model artifact to {model_path}")
    return payload


if __name__ == "__main__":
    train_salary_linear_regression()
