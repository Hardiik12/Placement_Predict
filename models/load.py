"""
models/load.py
--------------
Model loader and helper functions for loading saved model artifacts and making predictions.
"""

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "placement_model.joblib"
)

def load_prediction_model():
    if not MODEL_PATH.exists():
        # Fallback to train if missing
        from models.train_model import train_and_save
        train_and_save()
    return joblib.load(MODEL_PATH)


def get_feature_info():
    package = load_prediction_model()
    return {
        "numeric_cols": package.get("numeric_cols", []),
        "categorical_cols": package.get("categorical_cols", []),
        "categories": package.get("categories", {})
    }


def predict_student(student_data):
    package = load_prediction_model()

    model = package["model"]
    numeric_cols = package.get("numeric_cols", [])
    feature_names = package.get("feature_names", [])

    row = {}
    for feature in feature_names:
        value = student_data.get(feature, "")

        if feature in numeric_cols:
            row[feature] = (
                float(value) if value is not None and str(value).strip() != "" else float("nan")
            )
        else:
            row[feature] = str(value) if value is not None and str(value).strip() != "" else None

    sample = pd.DataFrame([row], columns=feature_names)

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "prediction": str(prediction),
        "confidence": confidence
    }

from ml.data_loader import load_data, clean_data
from ml.trainer import load_trained_model, predict_single, train_model

__all__ = [
    "load_data",
    "clean_data",
    "load_prediction_model",
    "get_feature_info",
    "predict_student",
    "load_trained_model",
    "predict_single",
    "train_model",
]
