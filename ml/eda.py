"""
ml/eda.py
---------
Exploratory Data Analysis helpers: summary statistics and saved chart images.

Charts are written to static/charts/ as PNG files so they can be served
directly by Flask/Jinja via url_for('static', filename='charts/<name>.png').
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data_loader import NUMERIC_COLS, TARGET_COL

sns.set_theme(style="whitegrid")

CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


def _ensure_charts_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Numeric describe() table + categorical value counts, returned as
    plain Python / JSON-friendly structures for rendering in a template.
    """
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    describe_df = df[numeric_cols].describe().T.round(2)

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_summary = {
        col: df[col].value_counts().head(10).to_dict() for col in categorical_cols
    }

    placement_rate = None
    if TARGET_COL in df.columns:
        placement_rate = round(df[TARGET_COL].mean() * 100, 2)

    return {
        "numeric_describe": describe_df.to_dict(orient="index"),
        "categorical_summary": cat_summary,
        "placement_rate": placement_rate,
        "correlation_with_target": _correlation_with_target(df, numeric_cols),
    }


def _correlation_with_target(df: pd.DataFrame, numeric_cols):
    if TARGET_COL not in df.columns:
        return {}
    corr = df[numeric_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    corr = corr.sort_values(ascending=False).round(3)
    return corr.to_dict()


def generate_charts(df: pd.DataFrame) -> list:
    """
    Generate a standard set of EDA charts and save them as PNGs under
    static/charts/. Returns the list of generated filenames (relative to
    static/charts/) so templates can build <img> tags.
    """
    _ensure_charts_dir()
    filenames = []

    # 1. Target class distribution
    if TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x=TARGET_COL, data=df, hue=TARGET_COL, ax=ax, palette="viridis", legend=False)
        ax.set_title("Placement Status Distribution")
        ax.set_xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
        fname = "placement_distribution.png"
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    # 2. CGPA distribution by placement status
    if "CGPA" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(
            data=df, x="CGPA", hue=TARGET_COL, kde=True, ax=ax, palette="viridis",
            element="step",
        )
        ax.set_title("CGPA Distribution by Placement Status")
        fname = "cgpa_distribution.png"
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    # 3. Correlation heatmap (numeric features)
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=False)
        ax.set_title("Feature Correlation Heatmap")
        fname = "correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    # 4. Placement rate by CollegeTier
    if "CollegeTier" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        rate = df.groupby("CollegeTier")[TARGET_COL].mean().sort_values(ascending=False)
        sns.barplot(x=rate.index, y=rate.values, hue=rate.index, ax=ax, palette="viridis", legend=False)
        ax.set_title("Placement Rate by College Tier")
        ax.set_ylabel("Placement Rate")
        fname = "placement_by_college_tier.png"
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    # 5. Internships vs Placement
    if "Internships" in df.columns and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        rate = df.groupby("Internships")[TARGET_COL].mean()
        sns.barplot(x=rate.index, y=rate.values, hue=rate.index, ax=ax, palette="viridis", legend=False)
        ax.set_title("Placement Rate by Number of Internships")
        ax.set_xlabel("Internships")
        ax.set_ylabel("Placement Rate")
        fname = "placement_by_internships.png"
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    # 6. Add Linear Regression & DBSCAN charts if present
    for extra_fname in [
        "simple_linear_regression.png",
        "multiple_linear_regression.png",
        "dbscan_clusters.png",
        "dbscan_pca_projection.png",
    ]:
        if os.path.exists(os.path.join(CHARTS_DIR, extra_fname)):
            filenames.append(extra_fname)

    return filenames


if __name__ == "__main__":
    from ml.data_loader import load_data, clean_data

    raw = load_data()
    cleaned = clean_data(raw)
    stats = get_summary_stats(cleaned)
    print("Placement rate:", stats["placement_rate"])
    charts = generate_charts(cleaned)
    print("Generated charts:", charts)
