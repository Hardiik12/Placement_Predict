"""
models/feature_engg.py
----------------------
Feature engineering module providing normalization, standardization, and outlier handling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize(df: pd.DataFrame, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols_all = df.select_dtypes(include=np.number).columns
    scale_cols = [c for c in numeric_cols_all if c not in exclude_cols]

    numeric_df = df[scale_cols].copy()
    keep_df = df.drop(columns=scale_cols).copy()

    # Min-Max Normalization
    min_max_scaler = MinMaxScaler()
    normalized_array = min_max_scaler.fit_transform(numeric_df)
    normalized_numeric_df = pd.DataFrame(
        normalized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    normalized_df = pd.concat([keep_df, normalized_numeric_df], axis=1)[df.columns]

    # Standardization
    standard_scaler = StandardScaler()
    standardized_array = standard_scaler.fit_transform(numeric_df)
    standardized_numeric_df = pd.DataFrame(
        standardized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    standardized_df = pd.concat([keep_df, standardized_numeric_df], axis=1)[df.columns]

    return normalized_df, standardized_df


def outlier(df: pd.DataFrame) -> pd.DataFrame:
    if "Salary Package" not in df.columns:
        return df
    df = df.copy()
    q1 = df["Salary Package"].quantile(0.25)
    q3 = df["Salary Package"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df["Salary Package"] = df["Salary Package"].clip(upper=upper)
    return df
