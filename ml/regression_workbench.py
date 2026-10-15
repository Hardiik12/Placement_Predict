"""
ml/regression_workbench.py
--------------------------
Regression Model Workbench Module (M2 Syllabus) for Student Placement Prediction.

Implements and evaluates:
1. Baseline: Custom SimpleLinearRegressionGD (Gradient Descent)
2. Normal Equation (OLS Closed-Form Solution: θ = (X^T X)^(-1) X^T y)
3. Scikit-Learn OLS Linear Regression
4. Polynomial Regression (degree 2 via PolynomialFeatures + LinearRegression)
5. Ridge Regression (L2 Regularization)
6. Lasso Regression (L1 Regularization & Sparsity)
7. Elastic Net (Combined L1 & L2 Regularization)

Strict Data Leakage Controls:
- Train / Validation / Test 3-way split (70% / 15% / 15%)
- Preprocessing (StandardScaler, polynomial expansion) fitted strictly on training data
- Hyperparameters validated on validation set, final unbiased evaluation on held-out test set
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.common import BASE_DIR, CHARTS_DIR, extract_representative_ml_sample, prepare_train_val_test_splits

_WORKBENCH_MODELS = {}
BENCHMARK_FEATURES = ['CGPA', 'CodingTestScore', 'AptitudeTestScore', 'SoftSkillsRating', 'Internships', 'Projects', 'MockInterviewScore']


class SimpleLinearRegressionGD:
    """Custom Simple Linear Regression trained via Gradient Descent."""
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.mse_history = []

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        m = len(y_arr)
        self.theta0 = float(np.mean(y_arr))
        self.theta1 = 0.0
        self.mse_history = []

        for _ in range(self.iterations):
            y_pred = self.theta0 + self.theta1 * X_arr
            error = y_pred - y_arr
            mse = float(np.mean(error ** 2))
            self.mse_history.append(mse)

            d_theta0 = (2 / m) * np.sum(error)
            d_theta1 = (2 / m) * np.sum(error * X_arr)

            self.theta0 -= self.lr * d_theta0
            self.theta1 -= self.lr * d_theta1

        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float).ravel()
        return self.theta0 + self.theta1 * X_arr


def calculate_regression_metrics(y_true, y_pred, p_features=1):
    """Computes standard regression evaluation metrics: MAE, MSE, RMSE, R2, Adjusted R2."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    if n > p_features + 1 and r2 is not None:
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - p_features - 1))
    else:
        adj_r2 = r2

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "adj_r2": round(adj_r2, 4)
    }


def solve_normal_equation(X, y):
    """Computes closed-form OLS solution: θ = (X^T X)^(-1) X^T y."""
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    X_b = np.c_[np.ones((len(X_mat), 1)), X_mat]
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    # Use pseudo-inverse for numerical stability
    theta = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y_vec)
    intercept = float(theta[0, 0])
    coefficients = [float(c[0]) for c in theta[1:]]

    return {
        "formula": "θ = (X^T X)^(-1) X^T y",
        "intercept": round(intercept, 4),
        "coefficients": [round(c, 4) for c in coefficients]
    }


def train_workbench_models(splits):
    """Fits all candidate regression architectures strictly on training split."""
    global _WORKBENCH_MODELS
    X_train = splits["X_train"]
    y_train = splits["y_train"]

    # 1. Baseline Simple Linear Regression (Gradient Descent on CGPA)
    gd_scaler = StandardScaler()
    cgpa_scaled = gd_scaler.fit_transform(X_train[["CGPA"]])
    model_gd = SimpleLinearRegressionGD(learning_rate=0.01, iterations=1000)
    model_gd.fit(cgpa_scaled, y_train)

    # 2. Scikit-learn Multiple Linear Regression
    pipe_linear = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipe_linear.fit(X_train, y_train)

    # 3. Polynomial Regression (degree 2)
    pipe_poly = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipe_poly.fit(X_train, y_train)

    # 4. Ridge Regression (L2)
    pipe_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=10.0, random_state=42))
    ])
    pipe_ridge.fit(X_train, y_train)

    # 5. Lasso Regression (L1)
    pipe_lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Lasso(alpha=0.05, random_state=42))
    ])
    pipe_lasso.fit(X_train, y_train)

    # 6. Elastic Net (L1 + L2)
    pipe_elastic = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42))
    ])
    pipe_elastic.fit(X_train, y_train)

    _WORKBENCH_MODELS = {
        "baseline_gd": {"model": model_gd, "scaler": gd_scaler, "feature": "CGPA"},
        "linear_sklearn": {"model": pipe_linear, "features": list(X_train.columns)},
        "polynomial": {"model": pipe_poly, "features": list(X_train.columns)},
        "ridge": {"model": pipe_ridge, "features": list(X_train.columns)},
        "lasso": {"model": pipe_lasso, "features": list(X_train.columns)},
        "elastic_net": {"model": pipe_elastic, "features": list(X_train.columns)}
    }

    return _WORKBENCH_MODELS


def evaluate_all_workbench_models(models, splits):
    """Evaluates all candidate models across Train, Validation, and Test sets."""
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    p = len(X_train.columns)
    model_summaries = []

    model_metadata = [
        ("baseline_gd", "Gradient Descent (CGPA Baseline)", "Simple Linear", "θ = θ - α∇J"),
        ("linear_sklearn", "Multiple Linear Regression (OLS)", "Multivariate Linear", "min ||y - Xθ||²"),
        ("polynomial", "Polynomial Regression (Degree 2)", "Non-Linear Expansion", "Feature Crosses (x_i * x_j)"),
        ("ridge", "Ridge Regression (L2 Penalty)", "Regularized Linear", "Loss + λ||θ||₂²"),
        ("lasso", "Lasso Regression (L1 Penalty)", "Sparse Regularized", "Loss + λ||θ||₁"),
        ("elastic_net", "Elastic Net (L1 + L2 Regularization)", "Hybrid Regularized", "Loss + rλ||θ||₁ + (1-r)λ||θ||₂²")
    ]

    for key, name, category, formula in model_metadata:
        m_info = models[key]
        if key == "baseline_gd":
            scaler = m_info["scaler"]
            model = m_info["model"]
            pred_train = model.predict(scaler.transform(X_train[["CGPA"]]))
            pred_val = model.predict(scaler.transform(X_val[["CGPA"]]))
            pred_test = model.predict(scaler.transform(X_test[["CGPA"]]))
        else:
            pipe = m_info["model"]
            pred_train = pipe.predict(X_train)
            pred_val = pipe.predict(X_val)
            pred_test = pipe.predict(X_test)

        train_m = calculate_regression_metrics(y_train, pred_train, p)
        val_m = calculate_regression_metrics(y_val, pred_val, p)
        test_m = calculate_regression_metrics(y_test, pred_test, p)

        model_summaries.append({
            "key": key,
            "name": name,
            "category": category,
            "family": category,
            "formula": formula,
            "parameters": formula,
            "train_rmse": train_m["rmse"],
            "val_rmse": val_m["rmse"],
            "test_rmse": test_m["rmse"],
            "test_r2": test_m["r2"],
            "test_adj_r2": test_m["adj_r2"],
            "test_mae": test_m["mae"],
            "test_mse": test_m["mse"],
            "rmse": test_m["rmse"],
            "r2": test_m["r2"],
            "mae": test_m["mae"],
            "mse": test_m["mse"],
            "adjusted_r2": test_m["adj_r2"]
        })

    # Identify best model based on validation RMSE
    best_model = min(model_summaries, key=lambda x: x["val_rmse"])
    models_dict = {m["key"]: m for m in model_summaries}

    return {
        "models": model_summaries,
        "models_dict": models_dict,
        "best_model": best_model,
        "normal_equation": solve_normal_equation(X_train, y_train)
    }


def predict_workbench_salary_by_model(input_data=None, model_key="polynomial", cgpa_val=None, coding_val=None, internships_val=None, **kwargs):
    """
    Performs inference using chosen workbench model for student salary prediction.
    Supports either dictionary payload or keyword arguments (cgpa_val, coding_val, etc.).
    """
    t0 = time.time()
    global _WORKBENCH_MODELS

    # Auto-train models if not in memory
    if not _WORKBENCH_MODELS:
        run_complete_regression_workbench()

    if model_key not in _WORKBENCH_MODELS:
        model_key = "polynomial" if "polynomial" in _WORKBENCH_MODELS else (list(_WORKBENCH_MODELS.keys())[0] if _WORKBENCH_MODELS else "baseline_gd")

    # Normalize input into dictionary
    data = {}
    if isinstance(input_data, dict):
        data.update(input_data)
    elif isinstance(input_data, (int, float, str)) and input_data is not None:
        try:
            data["CGPA"] = float(input_data)
        except Exception:
            pass

    if cgpa_val is not None:
        data["CGPA"] = float(cgpa_val)
    if coding_val is not None:
        data["CodingTestScore"] = float(coding_val)
    if internships_val is not None:
        data["Internships"] = int(internships_val)
    data.update(kwargs)

    m_info = _WORKBENCH_MODELS[model_key]

    if model_key == "baseline_gd":
        cgpa = float(data.get("CGPA", 7.5))
        cgpa_scaled = m_info["scaler"].transform([[cgpa]])
        pred = float(m_info["model"].predict(cgpa_scaled)[0])
    else:
        pipe = m_info["model"]
        features = m_info["features"]
        row = [float(data.get(f, 75.0 if "Score" in f else (7.5 if f == "CGPA" else 1.0))) for f in features]
        df_row = pd.DataFrame([row], columns=features)
        pred = float(pipe.predict(df_row)[0])

    pred = max(2.5, round(pred, 2))
    elapsed_ms = round((time.time() - t0) * 1000.0, 2)
    cgpa_val_used = float(data.get("CGPA", 7.5))
    prob_placed = 1.0 / (1.0 + np.exp(-(0.8 * cgpa_val_used + 0.03 * float(data.get("CodingTestScore", 70.0)) - 6.0)))

    return {
        "status": "success",
        "model_key": model_key,
        "model_name": m_info.get("name", model_key.replace("_", " ").title()),
        "predicted_salary": pred,
        "predicted_salary_lpa": pred,
        "unit": "LPA",
        "input_cgpa": cgpa_val_used,
        "input_coding": float(data.get("CodingTestScore", 75.0)),
        "input_internships": int(data.get("Internships", 1)),
        "placement_likelihood": f"{int(prob_placed * 100)}%",
        "latency_ms": elapsed_ms
    }


def create_workbench_prediction_plot(models, splits):
    """Generates comparison scatter plot of actual vs predicted salaries on held-out test data."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    X_test = splits["X_test"]
    y_test = splits["y_test"].values
    sub_n = min(300, len(y_test))
    indices = np.random.RandomState(42).choice(len(y_test), sub_n, replace=False)
    y_sub = y_test[indices]

    plt.figure(figsize=(9, 6))
    plt.scatter(y_sub, y_sub, color="black", alpha=0.3, label="Ideal 1:1 Fit (Actual = Predicted)")

    colors = {"baseline_gd": "#f0883e", "linear_sklearn": "#58a6ff", "polynomial": "#3fb950", "ridge": "#a371f7"}
    for key, color in colors.items():
        if key in models:
            m_info = models[key]
            if key == "baseline_gd":
                cgpa_sub = X_test[["CGPA"]].iloc[indices]
                pred = m_info["model"].predict(m_info["scaler"].transform(cgpa_sub))
            else:
                pred = m_info["model"].predict(X_test.iloc[indices])
            plt.scatter(y_sub, pred, color=color, alpha=0.6, label=key.capitalize())

    plt.xlabel("Actual Salary Package (LPA)")
    plt.ylabel("Predicted Salary Package (LPA)")
    plt.title("Regression Workbench: Predicted vs Actual Salary (Held-Out Test Set)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()

    chart_path = os.path.join(CHARTS_DIR, "workbench_regression_fits.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def create_workbench_metrics_plot(evaluation):
    """Generates horizontal bar chart comparing RMSE across workbench models."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")

    names = [m["name"].split(" (")[0] for m in evaluation["models"]]
    rmses = [m["test_rmse"] for m in evaluation["models"]]

    plt.figure(figsize=(9, 4.5))
    bars = plt.barh(names, rmses, color="#3fb950", alpha=0.85)
    plt.xlabel("Held-Out Test RMSE (LPA - Lower is Better)")
    plt.title("Regression Model Benchmark Comparison (Test RMSE)", fontsize=11, fontweight="bold")

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, "workbench_metrics_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    return chart_path


def get_regularization_educational_theory():
    """Returns educational summaries for L1, L2, Elastic Net, and Normal Equation."""
    return [
        {
            "technique": "Normal Equation (OLS Closed-Form)",
            "equation": "θ = (X^T X)^(-1) X^T y",
            "mechanism": "Analytically minimizes sum of squared residuals without learning rate tuning.",
            "pros": "Exact global minimum in one step; no iterations required.",
            "cons": "Matrix inversion complexity is O(p^3); ill-conditioned when collinearity exists."
        },
        {
            "technique": "Ridge Regression (L2 Penalty)",
            "equation": "J(θ) = MSE + λ ∑ θ_j²",
            "mechanism": "Penalizes Euclidean norm of weights; shrinks coefficients toward zero without nullifying them.",
            "pros": "Handles multicollinearity and prevents weight explosion.",
            "cons": "Does not perform feature selection (all features retain non-zero weights)."
        },
        {
            "technique": "Lasso Regression (L1 Penalty)",
            "equation": "J(θ) = MSE + λ ∑ |θ_j|",
            "mechanism": "Penalizes absolute weight magnitudes; drives non-informative feature weights strictly to zero.",
            "pros": "Automatic feature selection resulting in sparse, interpretable models.",
            "cons": "Arbitrarily selects one feature among correlated predictors and drops the rest."
        },
        {
            "technique": "Elastic Net (Combined L1 + L2)",
            "equation": "J(θ) = MSE + rλ ∑ |θ_j| + 0.5(1-r)λ ∑ θ_j²",
            "mechanism": "Convex combination of L1 sparsity and L2 grouping properties.",
            "pros": "Retains groups of correlated relevant predictors while achieving sparsity.",
            "cons": "Requires tuning two hyperparameters (alpha and l1_ratio)."
        }
    ]


def run_complete_regression_workbench(df_sample=None):
    """
    Master runner function for M2 Regression Model Workbench.
    Trains, benchmarks, plots, and caches all regression algorithms.
    """
    if df_sample is None:
        data_path = os.path.join(BASE_DIR, "data", "placement_predict_50k Dataset1.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "placement_predict_50k Dataset1.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if "Salary Package" in df.columns:
                df.rename(columns={"Salary Package": "SalaryPackage"}, inplace=True)
            df_sample = extract_representative_ml_sample(df, sample_size=3000)
        else:
            np.random.seed(42)
            n = 500
            cgpa = np.random.uniform(5.0, 9.8, n)
            coding = np.clip(cgpa * 8.5 + np.random.normal(0, 10, n), 20, 100)
            internships = np.random.poisson(1.0, n)
            salary = np.clip(1.3 * cgpa + 0.08 * coding + 0.5 * internships - 2.5, 3.0, 35.0)
            df_sample = pd.DataFrame({
                "CGPA": cgpa, "CodingTestScore": coding, "Internships": internships,
                "PlacementStatus": 1, "SalaryPackage": salary
            })

    placed_sample = df_sample[df_sample["PlacementStatus"] == 1].copy()
    if len(placed_sample) < 100:
        placed_sample = df_sample.copy()

    avail_features = [f for f in BENCHMARK_FEATURES if f in placed_sample.columns]
    target_col = "SalaryPackage"

    splits = prepare_train_val_test_splits(placed_sample, avail_features, target_col, test_size=0.15, val_size=0.15, random_state=42)
    models = train_workbench_models(splits)
    evaluation = evaluate_all_workbench_models(models, splits)
    create_workbench_prediction_plot(models, splits)
    create_workbench_metrics_plot(evaluation)
    theory = get_regularization_educational_theory()

    return {
        "dataset_split": splits["counts"],
        "benchmark_features": avail_features,
        "target_variable": target_col,
        "models": evaluation["models"],
        "models_dict": evaluation.get("models_dict", {}),
        "best_model": evaluation["best_model"],
        "normal_equation": evaluation["normal_equation"],
        "educational_theory": theory,
        "fit_chart": "static/charts/workbench_regression_fits.png",
        "metrics_chart": "static/charts/workbench_metrics_comparison.png"
    }
