"""
ml/outliers.py
--------------
Outlier Analysis & Treatment Module (CO1 Syllabus) for Student Placement Prediction.

Provides modular functions for IQR calculations (Q1, Q2/Median, Q3, IQR, Lower/Upper Bounds),
Boxplot visualization generation, Outlier Removal impact evaluation, Retain Outliers baseline,
Winsorization capping, and multi-treatment comparative analysis.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ml.common import CHARTS_DIR

TARGET_NUMERICAL_COLS = ['CGPA', 'CodingTestScore', 'AptitudeTestScore', 'SalaryPackage']


def calculate_iqr_statistics(df):
    """
    Computes Q1 (25%), Q2/Median (50%), Q3 (75%), IQR, Lower Bound, Upper Bound,
    Outlier Count, and Outlier Percentage for numerical academic features.
    """
    total_rows = len(df)
    iqr_results = []

    # Standardize column name if needed
    col_mapping = {"Salary Package": "SalaryPackage"}
    df_eval = df.rename(columns=col_mapping)

    avail_cols = [c for c in TARGET_NUMERICAL_COLS if c in df_eval.columns]

    for col in avail_cols:
        series = df_eval[col].dropna()
        q25, q50, q75 = np.percentile(series, [25, 50, 75])
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        outlier_cnt = int(((series < lower) | (series > upper)).sum())
        outlier_pct = round((outlier_cnt / total_rows) * 100.0, 2)

        iqr_results.append({
            "variable": col,
            "q1": round(float(q25), 2),
            "median": round(float(q50), 2),
            "q3": round(float(q75), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "potential_outliers": outlier_cnt,
            "outlier_pct": outlier_pct
        })

    return iqr_results


def create_boxplots(df):
    """
    Generates a 2x2 subplot boxplot chart figure saved as static/charts/outlier_boxplots.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    df_eval = df.rename(columns={"Salary Package": "SalaryPackage"})
    sample_df = df_eval.sample(n=min(5000, len(df_eval)), random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Student Numerical Feature Outlier Boxplots (IQR Method)", fontsize=14, fontweight='bold', y=0.98)

    plots_config = [
        ('CGPA', 'CGPA (0 - 10)', axes[0, 0], '#2f6f4f'),
        ('CodingTestScore', 'Coding Test Score (0 - 100)', axes[0, 1], '#3fb950'),
        ('AptitudeTestScore', 'Aptitude Test Score (0 - 100)', axes[1, 0], '#58a6ff'),
        ('SalaryPackage', 'Salary Package (LPA)', axes[1, 1], '#a371f7')
    ]

    for col, label, ax, color in plots_config:
        if col in sample_df.columns:
            sns.boxplot(x=sample_df[col].dropna(), ax=ax, color=color, fliersize=3, boxprops=dict(alpha=0.85))
            ax.set_title(f"Boxplot: {label}", fontsize=11, fontweight='bold')
            ax.set_xlabel(label)
            ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(CHARTS_DIR, "outlier_boxplots.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return plot_path


def evaluate_outlier_treatments(df, iqr_table):
    """
    Evaluates and compares the 3 outlier treatment strategies:
    1. Remove Outliers (IQR Filtering)
    2. Retain Outliers (Base Clean Dataset)
    3. Winsorization (Lower & Upper Bound Capping)
    """
    df_eval = df.rename(columns={"Salary Package": "SalaryPackage"})
    total_rows = len(df_eval)

    # 1. Remove Outliers (Analytical Evaluation)
    condition = pd.Series(True, index=df_eval.index)
    for row in iqr_table:
        col = row["variable"]
        if col in df_eval.columns:
            lower = row["lower_bound"]
            upper = row["upper_bound"]
            condition = condition & (df_eval[col] >= lower) & (df_eval[col] <= upper)

    rows_after_removal = int(condition.sum())
    removed_cnt = total_rows - rows_after_removal
    retention_pct = round((rows_after_removal / total_rows) * 100.0, 2)

    remove_impact = {
        "original_clean_records": total_rows,
        "records_after_removal": rows_after_removal,
        "records_removed": removed_cnt,
        "retention_pct": retention_pct
    }

    # 2. Retain Outliers (Baseline)
    retain_impact = {
        "original_clean_records": total_rows,
        "records_retained": total_rows,
        "records_removed": 0,
        "retention_pct": 100.0
    }

    # 3. Winsorization Comparison
    winsorize_comparison = []
    for row in iqr_table:
        col = row["variable"]
        if col in df_eval.columns:
            lower = row["lower_bound"]
            upper = row["upper_bound"]
            orig_series = df_eval[col].dropna()
            capped_series = np.clip(orig_series, lower, upper)

            winsorize_comparison.append({
                "variable": col,
                "mean_before": round(float(orig_series.mean()), 2),
                "mean_after": round(float(capped_series.mean()), 2),
                "median_before": round(float(orig_series.median()), 2),
                "median_after": round(float(capped_series.median()), 2),
                "std_before": round(float(orig_series.std()), 2),
                "std_after": round(float(capped_series.std()), 2)
            })

    method_comparison_table = [
        {
            "method": "Remove Outliers (IQR Filtering)",
            "records_removed": f"{removed_cnt:,}",
            "data_preserved_pct": f"{retention_pct}%",
            "effect": "Eliminates all student profiles outside Q1 - 1.5*IQR and Q3 + 1.5*IQR bounds"
        },
        {
            "method": "Retain Outliers (Keep Original)",
            "records_removed": "0",
            "data_preserved_pct": "100.0%",
            "effect": "Preserves genuine high-tier super-dream packages and low-score anomalies"
        },
        {
            "method": "Winsorization (Bound Capping)",
            "records_removed": "0",
            "data_preserved_pct": "100.0%",
            "effect": "Caps extreme packages at upper bound without dropping any student records"
        }
    ]

    return {
        "remove_impact": remove_impact,
        "retain_impact": retain_impact,
        "winsorize_comparison": winsorize_comparison,
        "method_comparison_table": method_comparison_table
    }


def run_complete_outlier_analysis(df):
    """
    Master runner function for CO1 Outlier Analysis.
    Calculates IQR metrics, creates boxplots, and evaluates treatment strategies.
    """
    iqr_table = calculate_iqr_statistics(df)
    boxplots_path = create_boxplots(df)
    treatments = evaluate_outlier_treatments(df, iqr_table)

    return {
        "iqr_table": iqr_table,
        "boxplots_chart": "static/charts/outlier_boxplots.png",
        "treatments": treatments
    }
