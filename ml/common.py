"""
ml/common.py
------------
Student Placement Prediction Common Machine Learning Utilities and Shared Infrastructure.

Provides standardized, leakage-free data partitioning, feature extraction,
schema validation, and metric computation across M1-M6 modules.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPL_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)

REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Schema definitions for student placement inference validation
FEATURE_SCHEMA = {
    "CGPA": {"type": float, "min": 0.0, "max": 10.0, "required": True, "default": 7.5},
    "AttendancePercent": {"type": float, "min": 0.0, "max": 100.0, "required": False, "default": 85.0},
    "CodingTestScore": {"type": float, "min": 0.0, "max": 100.0, "required": False, "default": 75.0},
    "AptitudeTestScore": {"type": float, "min": 0.0, "max": 100.0, "required": False, "default": 75.0},
    "SoftSkillsRating": {"type": float, "min": 1.0, "max": 5.0, "required": False, "default": 4.0},
    "MockInterviewScore": {"type": float, "min": 0.0, "max": 100.0, "required": False, "default": 70.0},
    "Internships": {"type": int, "min": 0, "max": 10, "required": False, "default": 1},
    "Projects": {"type": int, "min": 0, "max": 20, "required": False, "default": 2},
    "Workshops": {"type": float, "min": 0.0, "max": 10.0, "required": False, "default": 2.0},
    "Stream": {"type": str, "required": False, "default": "CS"},
    "CollegeTier": {"type": str, "required": False, "default": "Tier 1"},
    "HistoryOfBacklogs": {"type": str, "required": False, "default": "No"},
}


def extract_representative_ml_sample(df, sample_size=10000, random_state=42):
    """
    Extracts a bounded, reproducible Pandas sample from the placement dataset
    with standard feature naming and target encodings.
    """
    pdf = df.copy()

    # Standardize column naming
    if "Salary Package" in pdf.columns:
        pdf.rename(columns={"Salary Package": "SalaryPackage"}, inplace=True)

    # Impute missing numeric values if any
    num_cols = pdf.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if pdf[c].isnull().any():
            pdf[c] = pdf[c].fillna(pdf[c].median())

    # Add engineered features
    if "CodingTestScore" in pdf.columns and "AptitudeTestScore" in pdf.columns:
        pdf["TechnicalComposite"] = (pdf["CodingTestScore"] * 0.6 + pdf["AptitudeTestScore"] * 0.4).round(2)
    else:
        pdf["TechnicalComposite"] = 75.0

    if "CGPA" in pdf.columns:
        pdf["CGPA_Tier"] = pd.cut(
            pdf["CGPA"],
            bins=[-np.inf, 6.0, 8.0, np.inf],
            labels=["Low", "Mid", "High"]
        )

    # Ensure binary target
    if "PlacementStatus" in pdf.columns:
        pdf["PlacementStatus"] = pdf["PlacementStatus"].astype(int)

    # Target: Salary Tier for multiclass classification
    if "SalaryPackage" in pdf.columns:
        pdf["SalaryTier"] = pd.cut(
            pdf["SalaryPackage"],
            bins=[-np.inf, 8.0, 15.0, np.inf],
            labels=[0, 1, 2]
        ).astype(int)

    if len(pdf) > sample_size:
        sample_df = pdf.sample(n=sample_size, random_state=random_state).copy()
    else:
        sample_df = pdf.copy()

    return sample_df


def prepare_train_val_test_splits(df_sample, feature_cols, target_col, test_size=0.15, val_size=0.15, random_state=42, stratify=False):
    """
    Creates strict 3-way partition: 70% Train, 15% Validation, 15% Test.
    Ensures zero leakage across sets.
    """
    avail_cols = [c for c in feature_cols if c in df_sample.columns]
    X = df_sample[avail_cols].copy()
    y = df_sample[target_col].copy()

    strat = y if (stratify and y.nunique() <= 10) else None

    # First split off test set (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Next split train and val from remainder
    val_ratio_of_rem = val_size / (1.0 - test_size)
    strat_rem = y_train_val if (stratify and y_train_val.nunique() <= 10) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_of_rem, random_state=random_state, stratify=strat_rem
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "total": len(X)
        }
    }


def validate_inference_payload(payload):
    """
    Validates single student profile input against FEATURE_SCHEMA.
    Returns cleaned dict with types coerced, or raises ValueError.
    """
    cleaned = {}
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary.")

    for feat, rules in FEATURE_SCHEMA.items():
        if feat in payload and str(payload[feat]).strip() != "":
            val = payload[feat]
            try:
                coerced = rules["type"](val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feat}' must be convertible to {rules['type'].__name__}.")

            if "min" in rules and coerced < rules["min"]:
                raise ValueError(f"Feature '{feat}' ({coerced}) below minimum allowed ({rules['min']}).")
            if "max" in rules and coerced > rules["max"]:
                raise ValueError(f"Feature '{feat}' ({coerced}) exceeds maximum allowed ({rules['max']}).")
            cleaned[feat] = coerced
        else:
            if rules.get("required", False):
                raise ValueError(f"Missing required feature: '{feat}'")
            cleaned[feat] = rules.get("default", 0.0)

    return cleaned
