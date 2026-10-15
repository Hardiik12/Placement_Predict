"""
ml/m3_trees.py
--------------
M3 — Tree-Based Models & Boosting Module for Student Placement Prediction.

Implements:
1. Decision Tree Regressor & Classifier
2. Split criteria: Gini Impurity, Entropy, Information Gain, MSE/Friedman-MSE
3. Stopping criteria & pre-pruning (max_depth, min_samples_split, min_samples_leaf)
4. Cost-complexity post-pruning (ccp_alpha path & effective alpha selection)
5. Random Forest with Out-of-Bag (OOB) score evaluation
6. AdaBoost (Adaptive sample re-weighting via SAMME/linear loss)
7. Gradient Boosting (GradientBoostingRegressor)
8. LightGBM & HistGradientBoosting (Histogram-based leaf-wise gradient boosting)
9. Feature Importance (MDI) & Permutation Feature Importance
10. Diagnostic charts saved to static/charts/m3_tree_benchmarks.png
"""

import os
from ml.common import BASE_DIR, CHARTS_DIR, extract_representative_ml_sample, prepare_train_val_test_splits
MPL_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_DIR
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml.common import CHARTS_DIR, extract_representative_ml_sample, prepare_train_val_test_splits

# Check optional LightGBM import
HAS_LIGHTGBM = False
LIGHTGBM_ERROR = None
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError as e:
    LIGHTGBM_ERROR = str(e)


def compute_tree_split_criteria(class_probabilities):
    """
    Computes mathematical split criteria for classification:
    - Gini Impurity: 1 - ∑ p_i²
    - Shannon Entropy: -∑ p_i * log2(p_i)
    """
    probs = np.array(class_probabilities, dtype=float)
    probs = probs[probs > 0]
    probs = probs / np.sum(probs)
    
    gini = 1.0 - float(np.sum(probs ** 2))
    entropy = -float(np.sum(probs * np.log2(probs)))
    return {
        "gini_impurity": round(gini, 4),
        "entropy": round(entropy, 4)
    }


def compute_cost_complexity_pruning_path(X_train, y_train):
    """
    Computes cost-complexity pruning path (ccp_alpha) for Decision Tree,
    returning candidate alphas and total leaf impurities.
    """
    clf = DecisionTreeRegressor(random_state=42, max_depth=6)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas
    impurities = path.impurities

    # Subsample representative alphas for visualization
    indices = np.linspace(0, len(alphas) - 1, min(6, len(alphas)), dtype=int)
    return {
        "total_alphas_found": len(alphas),
        "sample_alphas": [round(float(a), 5) for a in alphas[indices]],
        "sample_impurities": [round(float(imp), 4) for imp in impurities[indices]]
    }


def train_and_benchmark_tree_models(df_sample):
    """
    Trains and benchmarks:
    1. Decision Tree (Pre-pruned: max_depth=5)
    2. Random Forest (with OOB evaluation)
    3. AdaBoost
    4. Gradient Boosting
    5. HistGradientBoosting (LightGBM equivalent)
    
    Evaluates on held-out test split, measures training latency,
    and extracts MDI and Permutation feature importances.
    """
    feature_cols = [
        'CGPA', 'CodingTestScore', 'AptitudeTestScore',
        'SoftSkillsRating', 'MockInterviewScore', 'Internships', 'Projects'
    ]
    avail_features = [c for c in feature_cols if c in df_sample.columns]
    target_col = 'SalaryPackage' if 'SalaryPackage' in df_sample.columns else 'CGPA'

    splits = prepare_train_val_test_splits(df_sample, avail_features, target_col, random_state=42)
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']

    benchmarks = []
    trained_models = {}

    # 1. Decision Tree (Pre-pruned: max_depth=5)
    t0 = time.time()
    dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_time = (time.time() - t0) * 1000.0
    y_pred_dt = dt_model.predict(X_test)
    benchmarks.append({
        "key": "decision_tree",
        "name": "Decision Tree (Pre-Pruned, Depth 5)",
        "family": "Single Tree (CART)",
        "training_time_ms": round(dt_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_dt)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_dt))), 2),
        "r2": round(float(r2_score(y_test, y_pred_dt)), 4),
        "oob_score": None,
        "parameters": {"max_depth": 5, "min_samples_split": 20, "criterion": "squared_error"}
    })
    trained_models["decision_tree"] = dt_model

    # 2. Random Forest (50 estimators, OOB enabled)
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=50, max_depth=8, min_samples_leaf=5,
        oob_score=True, random_state=42, n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    rf_time = (time.time() - t0) * 1000.0
    y_pred_rf = rf_model.predict(X_test)
    benchmarks.append({
        "key": "random_forest",
        "name": "Random Forest (50 Trees, OOB)",
        "family": "Bagging Ensemble",
        "training_time_ms": round(rf_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_rf)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 2),
        "r2": round(float(r2_score(y_test, y_pred_rf)), 4),
        "oob_score": round(float(rf_model.oob_score_), 4),
        "parameters": {"n_estimators": 50, "max_depth": 8, "oob_score": True}
    })
    trained_models["random_forest"] = rf_model

    # 3. AdaBoost (Adaptive Boosting)
    t0 = time.time()
    ada_model = AdaBoostRegressor(n_estimators=30, learning_rate=0.1, random_state=42)
    ada_model.fit(X_train, y_train)
    ada_time = (time.time() - t0) * 1000.0
    y_pred_ada = ada_model.predict(X_test)
    benchmarks.append({
        "key": "adaboost",
        "name": "AdaBoost Regressor (Linear Loss)",
        "family": "Boosting (Adaptive Residuals)",
        "training_time_ms": round(ada_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_ada)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_ada))), 2),
        "r2": round(float(r2_score(y_test, y_pred_ada)), 4),
        "oob_score": None,
        "parameters": {"n_estimators": 30, "learning_rate": 0.1, "loss": "linear"}
    })
    trained_models["adaboost"] = ada_model

    # 4. Gradient Boosting
    t0 = time.time()
    gb_model = GradientBoostingRegressor(
        n_estimators=40, learning_rate=0.08, max_depth=4, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_time = (time.time() - t0) * 1000.0
    y_pred_gb = gb_model.predict(X_test)
    benchmarks.append({
        "key": "gradient_boosting",
        "name": "Gradient Boosting (Friedman MSE)",
        "family": "Boosting (Gradient Descent Loss)",
        "training_time_ms": round(gb_time, 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred_gb)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_gb))), 2),
        "r2": round(float(r2_score(y_test, y_pred_gb)), 4),
        "oob_score": None,
        "parameters": {"n_estimators": 40, "learning_rate": 0.08, "max_depth": 4}
    })
    trained_models["gradient_boosting"] = gb_model

    # 5. LightGBM / HistGradientBoosting (Histogram-based)
    lgb_status_note = ""
    if HAS_LIGHTGBM:
        t0 = time.time()
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_time = (time.time() - t0) * 1000.0
        y_pred_lgb = lgb_model.predict(X_test)
        lgb_status_note = "Native LightGBM Library (Leaf-wise GOSS)"
        benchmarks.append({
            "key": "lightgbm",
            "name": "LightGBM Regressor (Leaf-Wise)",
            "family": "Histogram-Based Gradient Boosting",
            "training_time_ms": round(lgb_time, 2),
            "mae": round(float(mean_absolute_error(y_test, y_pred_lgb)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_lgb))), 2),
            "r2": round(float(r2_score(y_test, y_pred_lgb)), 4),
            "oob_score": None,
            "parameters": {"num_leaves": 31, "learning_rate": 0.08}
        })
        trained_models["lightgbm"] = lgb_model
    else:
        t0 = time.time()
        hgb_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, random_state=42)
        hgb_model.fit(X_train, y_train)
        hgb_time = (time.time() - t0) * 1000.0
        y_pred_hgb = hgb_model.predict(X_test)
        lgb_status_note = "Scikit-Learn HistGradientBoosting (Native Histogram Binning fallback)"
        benchmarks.append({
            "key": "hist_gradient_boosting",
            "name": "HistGradientBoosting (Histogram-Binning)",
            "family": "Histogram-Based Gradient Boosting",
            "training_time_ms": round(hgb_time, 2),
            "mae": round(float(mean_absolute_error(y_test, y_pred_hgb)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_hgb))), 2),
            "r2": round(float(r2_score(y_test, y_pred_hgb)), 4),
            "oob_score": None,
            "parameters": {"max_iter": 100, "learning_rate": 0.08}
        })
        trained_models["hist_gradient_boosting"] = hgb_model

    # Feature Importance (MDI & Permutation)
    mdi_importances = dict(zip(avail_features, [round(float(v), 4) for v in rf_model.feature_importances_]))
    
    # Fast permutation importance on validation set
    perm_res = permutation_importance(rf_model, X_val, y_val, n_repeats=3, random_state=42)
    perm_importances = dict(zip(avail_features, [round(float(v), 4) for v in perm_res.importances_mean]))

    # Cost-complexity pruning path
    prune_path = compute_cost_complexity_pruning_path(X_train, y_train)

    # Chart generation
    chart_file = create_tree_benchmark_charts(benchmarks, mdi_importances, perm_importances)

    return {
        "benchmarks": benchmarks,
        "feature_importances": {
            "mdi_random_forest": mdi_importances,
            "permutation_importance": perm_importances
        },
        "cost_complexity_pruning": prune_path,
        "lightgbm_integration": {
            "native_lightgbm_available": HAS_LIGHTGBM,
            "status_note": lgb_status_note
        },
        "chart": chart_file,
        "trained_models": trained_models
    }


def create_tree_benchmark_charts(benchmarks, mdi_dict, perm_dict):
    """
    Generates comparison charts:
    1. R2 and RMSE across Tree families
    2. Training Latency (ms) comparison
    3. MDI vs Permutation Feature Importances
    Saves to static/charts/m3_tree_benchmarks.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')

    # 1. Model R2 Scores
    names = [b['key'].replace('_', ' ').title() for b in benchmarks]
    r2_scores = [b['r2'] for b in benchmarks]
    bar_colors = ['#58a6ff', '#3fb950', '#bc8cff', '#f0883e', '#d29922']

    bars = axes[0].bar(names, r2_scores, color=bar_colors[:len(names)], width=0.55)
    axes[0].set_title("Test R² Score Comparison", fontsize=11, fontweight='bold', color='#f0f6fc')
    axes[0].set_ylabel("R² Score", fontsize=9, color='#8b949e')
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis='x', rotation=25, labelsize=8)
    axes[0].grid(axis='y', linestyle='--', alpha=0.2, color='#30363d')
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8, color='#f0f6fc')

    # 2. Training Latency (ms)
    times = [b['training_time_ms'] for b in benchmarks]
    axes[1].barh(names, times, color='#58a6ff', height=0.55)
    axes[1].set_title("Training Latency (ms)", fontsize=11, fontweight='bold', color='#f0f6fc')
    axes[1].set_xlabel("Time (milliseconds)", fontsize=9, color='#8b949e')
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].grid(axis='x', linestyle='--', alpha=0.2, color='#30363d')

    # 3. Feature Importance (Random Forest MDI vs Permutation)
    features = list(mdi_dict.keys())
    mdi_vals = [mdi_dict[f] for f in features]
    perm_vals = [perm_dict.get(f, 0.0) for f in features]
    y_pos = np.arange(len(features))

    axes[2].barh(y_pos - 0.18, mdi_vals, height=0.35, color='#3fb950', label='MDI (Gini)')
    axes[2].barh(y_pos + 0.18, perm_vals, height=0.35, color='#bc8cff', label='Permutation')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(features, fontsize=8)
    axes[2].set_title("Feature Importance (Random Forest)", fontsize=11, fontweight='bold', color='#f0f6fc')
    axes[2].set_xlabel("Importance Score", fontsize=9, color='#8b949e')
    axes[2].legend(loc="lower right", facecolor='#161b22', edgecolor='#30363d', fontsize=8)
    axes[2].grid(axis='x', linestyle='--', alpha=0.2, color='#30363d')

    plt.suptitle("M3 Tree Models & Boosting Benchmarks (Held-Out Test Set)",
                 fontsize=13, fontweight='bold', color='#f0f6fc', y=1.02)
    plt.tight_layout()

    out_file = os.path.join(CHARTS_DIR, "m3_tree_benchmarks.png")
    plt.savefig(out_file, dpi=160, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return "m3_tree_benchmarks.png"


def run_complete_m3_trees(df_clean=None):
    """
    Orchestrates full M3 Tree-Based Models & Boosting benchmark pipeline.
    """
    if df_clean is not None:
        sample_df = extract_representative_ml_sample(df_clean, sample_size=1500, random_state=42)
    else:
        np.random.seed(42)
        n = 1000
        cgpa = np.random.uniform(5.0, 9.8, n)
        coding = np.clip(cgpa * 8.5 + np.random.normal(0, 10, n), 20, 100)
        aptitude = np.clip(cgpa * 8.0 + np.random.normal(0, 8, n), 25, 100)
        soft = np.clip(cgpa * 0.4 + np.random.normal(0, 0.5, n), 1.0, 5.0)
        interview = np.clip(cgpa * 8.0 + np.random.normal(0, 10, n), 20, 100)
        internships = np.random.poisson(1.0, n)
        projects = np.random.poisson(2.0, n)
        salary = np.clip(1.3 * cgpa + 0.08 * coding + 0.5 * internships - 2.5 + np.random.normal(0, 1, n), 3.0, 35.0)

        sample_df = pd.DataFrame({
            "CGPA": cgpa,
            "CodingTestScore": coding,
            "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft,
            "MockInterviewScore": interview,
            "Internships": internships,
            "Projects": projects,
            "SalaryPackage": salary
        })

    return train_and_benchmark_tree_models(sample_df)
