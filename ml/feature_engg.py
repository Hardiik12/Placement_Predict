"""
ml/feature_engg.py
------------------
Feature engineering and normalization module for placement dataset.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Add project root to sys.path if missing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.data_loader import clean_data, load_data


def normalization(df: pd.DataFrame):
    """
    Splits dataframe into train/test sets and normalises numeric features
    using MinMaxScaler fitted on training data only (preventing data leakage).
    """
    num_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "SoftSkillsRating",
        "Internships",
        "Projects",
        "Workshops",
    ]

    # Filter num_cols to those present in df
    num_cols = [c for c in num_cols if c in df.columns]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"] if "PlacementStatus" in df.columns else None,
    )

    # Avoid SettingWithCopyWarning by creating explicit copies
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = MinMaxScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return train_df, test_df, scaler


if __name__ == "__main__":
    raw_df = load_data()
    df = clean_data(raw_df)
    train_df, test_df, scaler = normalization(df)

    print("Normalized Train shape:", train_df.shape)
    print("Normalized Test shape:", test_df.shape)
    print("\nSample normalized numeric values (Train):")
    print(
        train_df[
            [
                "CGPA",
                "AttendancePercent",
                "CodingTestScore",
            ]
        ].head()
    )