"""
ml/data_loader.py
------------------
Loads and cleans the placement dataset used throughout the app.

Dataset: data/placement_predict_50k_Dataset1.csv  (50,000 rows x 32 cols)

Columns
-------
StudentID, Gender, City, CollegeTier, Stream, Specialisation, Hostel,
HistoryOfBacklogs, SGPA_Sem1..SGPA_Sem8, CGPA, AttendancePercent,
Internships, Projects, Workshops, Certifications, Publications,
AptitudeTestScore, SoftSkillsRating, CodingTestScore, MockInterviewScore,
ExtraCurricular, CGPA_Tier, PlacementStatus (target), IsAnomaly,
Salary Package
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
_possible_paths = [
    os.path.join(DATA_DIR, "placement_predict_50k Dataset1.csv"),
    os.path.join(DATA_DIR, "placement_predict_50k_Dataset1.csv"),
]
DATA_PATH = next((p for p in _possible_paths if os.path.exists(p)), _possible_paths[0])


TARGET_COL = "PlacementStatus"

ID_COLS = ["StudentID"]

# Columns that had missing values in the raw file (see EDA)
NUMERIC_WITH_NULLS = [
    "Workshops",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
]

CATEGORICAL_COLS = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier",
]

SGPA_COLS = [f"SGPA_Sem{i}" for i in range(1, 9)]

NUMERIC_COLS = SGPA_COLS + [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw placement dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file. Defaults to data/placement_predict_50k_Dataset1.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe, exactly as read from disk (dtypes coerced sensibly).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV in the data/ folder."
        )

    df = pd.read_csv(path)

    # Normalise column name with a space, so downstream code can use either
    df.rename(columns={"Salary Package": "SalaryPackage"}, inplace=True)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Quick metadata about the dataframe, used by the data_loading page.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_counts": missing.to_dict(),
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COL].value_counts().to_dict()
        if TARGET_COL in df.columns
        else {},
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - impute missing numeric columns with the column median
      - drop exact duplicate rows
      - strip whitespace from string/categorical columns

    Returns a new dataframe; does not mutate the input.
    """
    clean = df.copy()

    # Drop duplicates
    clean.drop_duplicates(inplace=True)

    # Trim whitespace on categorical/text columns
    obj_cols = clean.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        clean[col] = clean[col].astype(str).str.strip()

    # Median-impute numeric columns that had nulls in the raw data
    for col in NUMERIC_WITH_NULLS:
        if col in clean.columns and clean[col].isnull().any():
            median_val = clean[col].median()
            clean[col] = clean[col].fillna(median_val)

    return clean


def get_feature_target_split(df: pd.DataFrame):
    """
    Split a cleaned dataframe into (X, y) ready for modelling.
    Drops identifier / leakage-prone columns (StudentID, IsAnomaly,
    SalaryPackage) from the feature matrix.
    """
    drop_cols = [c for c in ID_COLS + ["IsAnomaly", "SalaryPackage", TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    return X, y


if __name__ == "__main__":
    # Simple smoke test when run directly: `python ml/data_loader.py`
    raw = load_data()
    print("Raw shape:", raw.shape)
    print(get_data_summary(raw))

    cleaned = clean_data(raw)
    print("Cleaned shape:", cleaned.shape)
    print("Remaining nulls:", int(cleaned.isnull().sum().sum()))
