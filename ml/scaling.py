"""
ml/scaling.py
-------------
Feature Scaling & Normalization Module (CO1 Syllabus) for Student Placement Prediction.

Provides modular implementations for:
1. Standardization / Z-score (StandardScaler: mean=0, std=1)
2. Min-Max Scaling (MinMaxScaler: bound to [0, 1])
3. Robust Scaling (RobustScaler: scaled by median & IQR)
4. Before/after distribution statistics
5. Strict Data Leakage Prevention (fitting strictly on training set only)
6. Model Selection Guide (models requiring scaling vs tree models invariant to scaling)
7. Comparative visualization generation
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from ml.common import CHARTS_DIR

TARGET_SCALING_COLS = ['CGPA', 'CodingTestScore', 'AptitudeTestScore', 'AttendancePercent']


def compute_distribution_stats(series, name=""):
    """
    Computes distribution summary statistics for a 1D numeric array or series.
    Returns: dict with mean, std, min, median, max, iqr, q25, q75.
    """
    arr = np.asarray(series).ravel()
    q25, median, q75 = np.percentile(arr, [25, 50, 75])
    iqr = q75 - q25
    return {
        "feature": name,
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "q25": round(float(q25), 4),
        "median": round(float(median), 4),
        "q75": round(float(q75), 4),
        "max": round(float(np.max(arr)), 4),
        "iqr": round(float(iqr), 4)
    }


def standard_scaling(train_df, test_df=None, feature_cols=None):
    """Standardizes numerical features using Z-score formula: z = (x - mean) / std."""
    if feature_cols is None:
        feature_cols = [c for c in TARGET_SCALING_COLS if c in train_df.columns]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]), columns=feature_cols, index=test_df.index) if test_df is not None else None

    params = {col: {"mean": round(float(m), 4), "scale": round(float(s), 4)} for col, m, s in zip(feature_cols, scaler.mean_, scaler.scale_)}

    return {
        "scaler_name": "StandardScaler (Z-score)",
        "formula": "z = (x - mean) / std",
        "fitted_parameters": params,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "stats_before": [compute_distribution_stats(train_df[c], c) for c in feature_cols],
        "stats_after": [compute_distribution_stats(train_scaled[c], c) for c in feature_cols]
    }


def minmax_scaling(train_df, test_df=None, feature_cols=None):
    """Min-Max scales features: x_scaled = (x - min) / (max - min)."""
    if feature_cols is None:
        feature_cols = [c for c in TARGET_SCALING_COLS if c in train_df.columns]

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]), columns=feature_cols, index=test_df.index) if test_df is not None else None

    params = {col: {"min": round(float(mn), 4), "max": round(float(mx), 4)} for col, mn, mx in zip(feature_cols, scaler.data_min_, scaler.data_max_)}

    return {
        "scaler_name": "MinMaxScaler",
        "formula": "x_norm = (x - min) / (max - min)",
        "fitted_parameters": params,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "stats_before": [compute_distribution_stats(train_df[c], c) for c in feature_cols],
        "stats_after": [compute_distribution_stats(train_scaled[c], c) for c in feature_cols]
    }


def robust_scaling(train_df, test_df=None, feature_cols=None):
    """RobustScaler using median and IQR: x_robust = (x - median) / IQR."""
    if feature_cols is None:
        feature_cols = [c for c in TARGET_SCALING_COLS if c in train_df.columns]

    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]), columns=feature_cols, index=test_df.index) if test_df is not None else None

    params = {col: {"center_median": round(float(c), 4), "scale_iqr": round(float(s), 4)} for col, c, s in zip(feature_cols, scaler.center_, scaler.scale_)}

    return {
        "scaler_name": "RobustScaler",
        "formula": "x_rob = (x - median) / IQR",
        "fitted_parameters": params,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "stats_before": [compute_distribution_stats(train_df[c], c) for c in feature_cols],
        "stats_after": [compute_distribution_stats(train_scaled[c], c) for c in feature_cols]
    }


def demonstrate_scaling_leakage(df):
    """Demonstrates data leakage when scaling before train/test split."""
    feature = "CGPA" if "CGPA" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    sub = df[[feature]].dropna().sample(n=min(5000, len(df)), random_state=42)

    # Contaminated pipeline (Leaky: fit on entire dataset)
    scaler_leaky = StandardScaler()
    scaler_leaky.fit(sub)
    leaky_mean = float(scaler_leaky.mean_[0])

    # Clean pipeline (Zero leakage: fit strictly on 80% train)
    train_part, test_part = train_test_split(sub, test_size=0.2, random_state=42)
    scaler_clean = StandardScaler()
    scaler_clean.fit(train_part)
    clean_mean = float(scaler_clean.mean_[0])

    return {
        "feature": feature,
        "leaky_fit_dataset": "Full Sample (Train + Test combined)",
        "leaky_mean": round(leaky_mean, 4),
        "clean_fit_dataset": "Train Split Only (80%)",
        "clean_mean": round(clean_mean, 4),
        "leakage_delta": round(abs(leaky_mean - clean_mean), 6),
        "explanation": "Fitting scalers on test data bleeds test distribution information into the model, generating overly optimistic validation scores."
    }


def get_scaler_comparison_matrix():
    """Comparison table of scaler properties."""
    return [
        {
            "scaler": "StandardScaler",
            "formula": "z = (x - μ) / σ",
            "bounded": "No (-∞ to +∞)",
            "outlier_sensitivity": "High (Mean & Std distorted by outliers)",
            "best_use_case": "Gradient Descent, Linear & Logistic Regression, PCA, K-Means"
        },
        {
            "scaler": "MinMaxScaler",
            "formula": "x_norm = (x - min) / (max - min)",
            "bounded": "Yes ([0, 1])",
            "outlier_sensitivity": "Very High (Compresses inliers into narrow band)",
            "best_use_case": "Neural Networks, KNN distance metrics, image intensity"
        },
        {
            "scaler": "RobustScaler",
            "formula": "x_rob = (x - Q2) / (Q3 - Q1)",
            "bounded": "No",
            "outlier_sensitivity": "Immune / Low (Median & IQR are outlier-resistant)",
            "best_use_case": "Heavy-tailed distributions, real-world skewed salary/metric data"
        }
    ]


def get_model_scaling_guidance():
    """Educational guide for which algorithms require feature scaling."""
    return [
        {"model_family": "Linear & Logistic Regression", "requires_scaling": "Yes", "reason": "Features on larger scales dominate gradient updates and bias regularisation penalties."},
        {"model_family": "K-Nearest Neighbors & K-Means", "requires_scaling": "Yes", "reason": "Euclidean distance calculations are completely distorted by unscaled feature magnitudes."},
        {"model_family": "Principal Component Analysis (PCA)", "requires_scaling": "Yes", "reason": "Variance-maximization projects onto whatever axis has arbitrarily larger units."},
        {"model_family": "Decision Trees & Random Forests", "requires_scaling": "No", "reason": "Greedy split criteria evaluate individual features independently via rank-order thresholds."}
    ]


def create_scaling_comparison_chart(df):
    """Generates 3-way distribution plot before and after scaling."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    feature = "CGPA" if "CGPA" in df.columns else "CodingTestScore"
    data = df[[feature]].dropna().sample(n=min(3000, len(df)), random_state=42)

    ss = StandardScaler().fit_transform(data).ravel()
    mms = MinMaxScaler().fit_transform(data).ravel()
    rs = RobustScaler().fit_transform(data).ravel()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Feature Scaling Comparison on {feature}", fontsize=13, fontweight="bold")

    sns.histplot(data[feature], kde=True, ax=axes[0, 0], color="#58a6ff")
    axes[0, 0].set_title(f"Original {feature}")

    sns.histplot(ss, kde=True, ax=axes[0, 1], color="#238636")
    axes[0, 1].set_title("StandardScaler (mean=0, std=1)")

    sns.histplot(mms, kde=True, ax=axes[1, 0], color="#f0883e")
    axes[1, 0].set_title("MinMaxScaler (0 to 1)")

    sns.histplot(rs, kde=True, ax=axes[1, 1], color="#a371f7")
    axes[1, 1].set_title("RobustScaler (median=0, IQR=1)")

    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, "scaling_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def run_complete_scaling_analysis(df):
    """Orchestrates complete CO1 Scaling Analysis."""
    avail_cols = [c for c in TARGET_SCALING_COLS if c in df.columns]
    train_df, test_df = train_test_split(df[avail_cols].dropna(), test_size=0.2, random_state=42)

    std_res = standard_scaling(train_df, test_df, avail_cols)
    mms_res = minmax_scaling(train_df, test_df, avail_cols)
    rob_res = robust_scaling(train_df, test_df, avail_cols)
    leakage = demonstrate_scaling_leakage(df)
    matrix = get_scaler_comparison_matrix()
    guidance = get_model_scaling_guidance()
    create_scaling_comparison_chart(df)

    return {
        "standard_scaler": std_res,
        "minmax_scaler": mms_res,
        "robust_scaler": rob_res,
        "leakage_demo": leakage,
        "comparison_matrix": matrix,
        "model_guidance": guidance,
        "chart_path": "static/charts/scaling_comparison.png"
    }
