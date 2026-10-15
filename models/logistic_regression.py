"""
models/logistic_regression.py
-----------------------------
Logistic regression modeling for binary placement prediction.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def simple_logistic_regression(df: pd.DataFrame) -> dict:
    if "CGPA" not in df.columns or "PlacementStatus" not in df.columns:
        return {"error": "Required columns CGPA / PlacementStatus missing"}

    X = df[["CGPA"]]
    y = df["PlacementStatus"]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    cm = confusion_matrix(test_y, pred_y).tolist()

    return {
        "intercept": round(float(model.intercept_[0]), 4),
        "coefficient": round(float(model.coef_[0][0]), 4),
        "accuracy": round(float(acc) * 100, 2),
        "confusion_matrix": cm,
    }
