"""
ml/trainer.py
------------
Machine Learning Model Training & Inference pipeline for Student Placement Prediction.

Saves model pipeline to models/placement_model.joblib
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data_loader import clean_data, get_feature_target_split, load_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "placement_model.joblib")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build column preprocessor for numeric and categorical columns."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return preprocessor


def train_model() -> dict:
    """
    Train a Placement Prediction Random Forest model, save the pipeline artifact,
    and return evaluation metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    raw_df = load_data()
    df = clean_data(raw_df)
    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_proba) * 100, 2),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    defaults = {}
    for col in num_cols:
        defaults[col] = float(X[col].median())
    for col in cat_cols:
        defaults[col] = str(X[col].mode()[0])

    joblib.dump(
        {
            "pipeline": pipeline,
            "model": pipeline,
            "metrics": metrics,
            "features": list(X.columns),
            "feature_names": list(X.columns),
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "categories": {
                col: [str(v) for v in X[col].dropna().unique().tolist()]
                for col in cat_cols
            },
            "defaults": defaults,
        },
        MODEL_PATH,
    )

    # Train Linear Regression for Salary Package Prediction
    try:
        from ml.linear_regression import train_salary_linear_regression
        train_salary_linear_regression()
    except Exception as e:
        print(f"Warning: Linear Regression training skipped: {e}")

    return metrics


def load_trained_model():
    """Load trained model dictionary artifact if present."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_single(input_data: dict) -> dict:
    """Predict placement probability and class for a single student profile."""
    model_obj = load_trained_model()
    if model_obj is None:
        # Auto-train model if artifact is missing
        train_model()
        model_obj = load_trained_model()

    model_or_pipeline = model_obj.get("pipeline") or model_obj.get("model")
    input_df = pd.DataFrame([input_data])

    defaults = model_obj.get("defaults") or model_obj.get("medians") or {}
    features = model_obj.get("features") or model_obj.get("feature_names") or []
    num_cols = model_obj.get("num_cols", [])

    # Fill missing expected features with type-appropriate defaults
    for col in features:
        if col not in input_df.columns:
            if col in defaults:
                input_df[col] = defaults[col]
            elif col in num_cols:
                input_df[col] = 0.0
            else:
                input_df[col] = "Unknown"

    # Align input dataframe columns
    input_df = input_df[features]

    pred_class = int(model_or_pipeline.predict(input_df)[0])
    pred_proba = round(float(model_or_pipeline.predict_proba(input_df)[0][1]) * 100, 1)

    predicted_salary = 0.0
    reg_path = os.path.join(MODEL_DIR, "salary_linear_regression.joblib")
    if os.path.exists(reg_path):
        reg_obj = joblib.load(reg_path)
        reg_model = reg_obj.get("model")
        reg_features = reg_obj.get("features", features)
        
        # Prepare numeric input dataframe for linear regression
        reg_input = pd.DataFrame([input_data])
        reg_defaults = reg_obj.get("medians", defaults)
        for col in reg_features:
            if col not in reg_input.columns:
                reg_input[col] = reg_defaults.get(col, 0.0)
        reg_input = reg_input[reg_features]

        predicted_salary = max(0.0, round(float(reg_model.predict(reg_input)[0]), 2))

    return {
        "placed": bool(pred_class == 1),
        "probability": pred_proba,
        "status": "Placed" if pred_class == 1 else "Not Placed",
        "predicted_salary": predicted_salary if pred_class == 1 else 0.0,
    }


if __name__ == "__main__":
    print("Training model...")
    results = train_model()
    print("Metrics:", results)
