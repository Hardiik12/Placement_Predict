"""
ml/missing_values.py
--------------------
Missing Value Handling Module (CO1 Syllabus) for Student Placement Prediction.

Provides modular functions for missing value auditing, MCAR/MAR/MNAR classification,
listwise row deletion, column deletion, mean & median imputation, KNN imputation,
Iterative / MICE-style imputation, missing indicator generation, and treatment comparisons.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    HAS_ITERATIVE = True
except ImportError:
    HAS_ITERATIVE = False

from ml.common import CHARTS_DIR


def audit_missing_values(df):
    """
    Performs column-wise missing value audit.
    Returns audit metrics dictionary including missing count, percentage, non-missing count, and data types.
    """
    total_rows = len(df)
    null_series = df.isnull().sum()
    audit_table = []

    for col, missing_cnt in null_series.items():
        missing_cnt = int(missing_cnt)
        missing_pct = round((missing_cnt / total_rows) * 100.0, 2)
        non_missing_cnt = total_rows - missing_cnt

        audit_table.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "non_missing_count": non_missing_cnt
        })

    return {
        "total_rows": total_rows,
        "audit_table": audit_table,
        "columns_with_missing": [r["column"] for r in audit_table if r["missing_count"] > 0]
    }


def classify_missing_mechanism():
    """
    Provides educational mechanisms (MCAR, MAR, MNAR) and interpretation
    for the student placement dataset.
    """
    return [
        {
            "mechanism": "MCAR (Missing Completely at Random)",
            "definition": "Missingness probability is completely independent of both observed and unobserved values.",
            "dataset_example": "A student accidentally missed submitting optional certification certificates or workshop attendance log due to technical web portal timeout.",
            "recommended_treatment": "Listwise deletion or mean/median imputation without introducing bias."
        },
        {
            "mechanism": "MAR (Missing at Random)",
            "definition": "Missingness depends systematically on observed data, but not on the missing value itself.",
            "dataset_example": "Freshmen or second-year students from Tier 3 colleges are less likely to report Mock Interview scores because mock drives were scheduled only for final-year candidates.",
            "recommended_treatment": "Conditional / Multiple Imputation (KNN or MICE) conditioned on CollegeTier and Stream."
        },
        {
            "mechanism": "MNAR (Missing Not at Random)",
            "definition": "Missingness depends directly on the unobserved value itself.",
            "dataset_example": "Students who scored very poorly on coding assessments chose not to report or save their coding score.",
            "recommended_treatment": "Explicit missingness indicator flag (Binary Dummy variable) + median imputation to preserve the informative negative signal."
        }
    ]


def delete_missing_rows(df):
    """Listwise deletion dropping any row with nulls."""
    initial = len(df)
    res = df.dropna()
    return {
        "treatment": "Listwise Row Deletion (dropna)",
        "rows_before": initial,
        "rows_after": len(res),
        "rows_dropped": initial - len(res),
        "retention_pct": round((len(res) / initial) * 100.0, 2)
    }


def delete_missing_columns(df, threshold_pct=30.0):
    """Drops columns exceeding missing threshold percentage."""
    total = len(df)
    drop_cols = [c for c in df.columns if (df[c].isnull().sum() / total) * 100.0 > threshold_pct]
    return {
        "treatment": f"Column Deletion (threshold > {threshold_pct}%)",
        "dropped_columns": drop_cols,
        "columns_retained": len(df.columns) - len(drop_cols)
    }


def mean_imputation(df, col):
    """Mean imputation for a single numerical column."""
    if col not in df.columns:
        return {}
    orig = df[col]
    mean_val = float(orig.mean())
    imputed = orig.fillna(mean_val)
    return {
        "column": col,
        "imputed_value": round(mean_val, 2),
        "mean_before": round(float(orig.dropna().mean()), 2),
        "mean_after": round(float(imputed.mean()), 2),
        "std_before": round(float(orig.dropna().std()), 2),
        "std_after": round(float(imputed.std()), 2)
    }


def median_imputation(df, col):
    """Median imputation for a single numerical column."""
    if col not in df.columns:
        return {}
    orig = df[col]
    med_val = float(orig.median())
    imputed = orig.fillna(med_val)
    return {
        "column": col,
        "imputed_value": round(med_val, 2),
        "median_before": round(float(orig.dropna().median()), 2),
        "median_after": round(float(imputed.median()), 2),
        "std_before": round(float(orig.dropna().std()), 2),
        "std_after": round(float(imputed.std()), 2)
    }


def knn_imputation(df, num_cols, n_neighbors=5):
    """KNN imputation using nearest neighbors."""
    sample = df[num_cols].dropna().sample(n=min(1000, len(df)), random_state=42).copy()
    # Artificially inject nulls to benchmark KNN recovery
    mask = np.random.RandomState(42).rand(*sample.shape) < 0.1
    corrupted = sample.mask(mask)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    recovered = pd.DataFrame(imputer.fit_transform(corrupted), columns=num_cols)

    mae_scores = {}
    for c in num_cols:
        true_vals = sample[c].values[mask[c]]
        pred_vals = recovered[c].values[mask[c]]
        if len(true_vals) > 0:
            mae_scores[c] = round(float(np.mean(np.abs(true_vals - pred_vals))), 3)
        else:
            mae_scores[c] = 0.0

    return {
        "method": f"KNN Imputation (k={n_neighbors})",
        "sample_size": len(sample),
        "mae_by_feature": mae_scores
    }


def iterative_imputation(df, num_cols):
    """MICE / IterativeImputer multivariate recovery."""
    if not HAS_ITERATIVE:
        return {"status": "IterativeImputer unavailable"}

    sample = df[num_cols].dropna().sample(n=min(1000, len(df)), random_state=42).copy()
    mask = np.random.RandomState(42).rand(*sample.shape) < 0.1
    corrupted = sample.mask(mask)

    imputer = IterativeImputer(max_iter=10, random_state=42)
    recovered = pd.DataFrame(imputer.fit_transform(corrupted), columns=num_cols)

    mae_scores = {}
    for c in num_cols:
        true_vals = sample[c].values[mask[c]]
        pred_vals = recovered[c].values[mask[c]]
        if len(true_vals) > 0:
            mae_scores[c] = round(float(np.mean(np.abs(true_vals - pred_vals))), 3)
        else:
            mae_scores[c] = 0.0

    return {
        "method": "MICE / Iterative Imputation",
        "sample_size": len(sample),
        "mae_by_feature": mae_scores
    }


def create_missing_indicators(df, cols):
    """Generates explicit binary missing indicator features."""
    indicators = {}
    for c in cols:
        if c in df.columns:
            indicators[f"{c}_is_missing"] = df[c].isnull().astype(int)
    return pd.DataFrame(indicators, index=df.index)


def compare_imputation_methods(df):
    """Compares baseline imputation techniques."""
    num_cols = ["Workshops", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore"]
    avail = [c for c in num_cols if c in df.columns]

    return [
        {
            "method": "Listwise Deletion",
            "pros": "Simple, no assumption on missing distribution",
            "cons": "Reduces sample size and power, introduces bias under MAR/MNAR",
            "leakage_risk": "None"
        },
        {
            "method": "Median Imputation",
            "pros": "Robust to extreme outliers, fast, deterministic",
            "cons": "Underestimates variance and distorts feature covariance",
            "leakage_risk": "Must compute median on Train fold only"
        },
        {
            "method": "KNN Imputation",
            "pros": "Captures local multi-attribute relationships",
            "cons": "Computationally expensive at O(N) inference time",
            "leakage_risk": "Must fit KNN tree on Train only"
        },
        {
            "method": "MICE (Iterative)",
            "pros": "Preserves multivariate distributions and standard errors",
            "cons": "Iterative chained equations require longer training time",
            "leakage_risk": "Chained regressors must fit on Train only"
        }
    ]


def create_missing_values_chart(audit_data):
    """Generates bar chart of missing percentages across columns."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    table = audit_data["audit_table"]
    cols = [r["column"] for r in table]
    pcts = [r["missing_pct"] for r in table]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(cols, pcts, color="#58a6ff", alpha=0.85)
    plt.xlabel("Missing Percentage (%)")
    plt.title("Missing Values Audit by Column", fontsize=12, fontweight="bold")
    plt.xlim(0, max(max(pcts, default=0) + 5, 10))

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, "missing_values_audit.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def run_complete_missing_value_analysis(df):
    """Master orchestrator for CO1 Missing Value Analysis."""
    audit = audit_missing_values(df)
    mechanisms = classify_missing_mechanism()
    row_del = delete_missing_rows(df)
    comp = compare_imputation_methods(df)
    create_missing_values_chart(audit)

    return {
        "audit": audit,
        "mechanisms": mechanisms,
        "row_deletion": row_del,
        "method_comparison": comp,
        "chart_path": "static/charts/missing_values_audit.png"
    }
