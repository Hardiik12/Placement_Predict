"""
models/linear_regression.py
---------------------------
Linear regression modeling using Gradient Descent and Scikit-Learn.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def gradient_descent(X, y, learning_rate=0.001, iterations=1000):
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = len(y)
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []
    for _ in range(iterations):
        y_pred = theta0 + theta1 * X
        error = y_pred - y
        mse = np.mean(error ** 2)
        mse_history.append(float(mse))
        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(error * X)
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1
    return theta0, theta1, mse_history


def simple_linear_regression_GD(df: pd.DataFrame) -> dict:
    if "CGPA" not in df.columns or "Salary Package" not in df.columns:
        return {"error": "Required columns CGPA / Salary Package missing"}
    X = df[["CGPA"]]
    y = df["Salary Package"]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    theta0, theta1, mse_history = gradient_descent(train_X, train_y, learning_rate=0.001, iterations=1000)
    y_pred = theta0 + theta1 * test_X["CGPA"].values
    mse = float(mean_squared_error(test_y, y_pred))
    r2 = float(r2_score(test_y, y_pred))
    return {
        "theta0": round(float(theta0), 4),
        "theta1": round(float(theta1), 4),
        "mse": round(mse, 4),
        "r2_score": round(r2 * 100, 2),
    }


def simple_linear_regression_sklearn(df: pd.DataFrame) -> dict:
    if "CGPA" not in df.columns or "Salary Package" not in df.columns:
        return {"error": "Required columns CGPA / Salary Package missing"}
    X = df[["CGPA"]]
    y = df["Salary Package"]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(train_X, train_y)
    lasso = Lasso(alpha=0.1).fit(train_X, train_y)
    ridge = Ridge(alpha=0.1).fit(train_X, train_y)

    pred_lr = model.predict(test_X)
    pred_lasso = lasso.predict(test_X)
    pred_ridge = ridge.predict(test_X)

    return {
        "intercept": round(float(model.intercept_), 4),
        "coefficient": round(float(model.coef_[0]), 4),
        "linear_mse": round(float(mean_squared_error(test_y, pred_lr)), 4),
        "lasso_mse": round(float(mean_squared_error(test_y, pred_lasso)), 4),
        "ridge_mse": round(float(mean_squared_error(test_y, pred_ridge)), 4),
        "r2_score": round(float(r2_score(test_y, pred_lr)) * 100, 2),
    }
