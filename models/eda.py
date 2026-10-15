"""
models/eda.py
-------------
Exploratory Data Analysis module providing summarize, stat, and visualize methods.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ml.eda import generate_charts, get_summary_stats

CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "charts")

def summarize(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=np.number)
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "columns": df.columns.tolist(),
        "head_html": df.head(10).to_html(classes="table data-table", index=False),
        "describe_html": df.describe().to_html(classes="table data-table"),
    }


def stat(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=np.number)
    stats_df = pd.DataFrame({
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Std Dev": numeric_df.std(),
        "Skewness": numeric_df.skew(),
        "Kurtosis": numeric_df.kurt(),
    })
    return {
        "stats_html": stats_df.to_html(classes="table data-table"),
        "corr_html": numeric_df.corr().to_html(classes="table data-table"),
    }


def visualize(df: pd.DataFrame) -> list:
    os.makedirs(CHART_DIR, exist_ok=True)
    charts = generate_charts(df)
    return [os.path.basename(c) for c in charts]
