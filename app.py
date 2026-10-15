"""
app.py
------
Flask Web Application for Student Placement Prediction and ML Analysis Dashboard.
"""

import os
import sys
from flask import Flask, jsonify, redirect, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.load import load_data, clean_data, load_trained_model, predict_single
from models import eda as eda_module
from models import feature_engg as fe_module
from models import linear_regression as lr_module
from models import logistic_regression as logit_module
from models.predict import predict_student, get_feature_info
from ml.data_loader import get_data_summary
from ml.eda import generate_charts, get_summary_stats

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

DATA_PATH = os.path.join(BASE_DIR, "data", "placement_predict_50k Dataset1.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset1.csv")

TARGET_COLS = ["PlacementStatus", "IsAnomaly", "Salary Package"]

_cache = {}


def get_df():
    if "df" not in _cache:
        raw_df = load_data(DATA_PATH)
        _cache["df"] = clean_data(raw_df)
    return _cache["df"]


@app.route("/")
def index():
    data_available = os.path.exists(DATA_PATH)
    return render_template("index.html", data_available=data_available, data_path=DATA_PATH)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        feature_info = get_feature_info()
    except Exception:
        feature_info = {"numeric_cols": [], "categorical_cols": [], "categories": {}}

    result = None
    error = None
    values = {}

    if request.method == "POST":
        values = request.form.to_dict()
        try:
            result = predict_single(values)
        except Exception:
            try:
                result = predict_student(values)
            except Exception as e2:
                error = f"Invalid input: {e2}"

    return render_template(
        "predict.html",
        numeric_cols=feature_info.get("numeric_cols", []),
        categorical_cols=feature_info.get("categorical_cols", []),
        categories=feature_info.get("categories", {}),
        values=values,
        form_data=values,
        result=result,
        error=error,
    )


@app.route("/data-loading")
def data_loading():
    """Detailed data loading and cleaning inspection page."""
    error = None
    raw_summary = None
    cleaned_summary = None
    preview_cols = []
    preview_rows = []

    try:
        raw_df = load_data(DATA_PATH)
        cleaned_df = clean_data(raw_df)
        raw_summary = get_data_summary(raw_df)
        cleaned_summary = get_data_summary(cleaned_df)
        preview_cols = cleaned_df.columns.tolist()
        preview_rows = cleaned_df.head(10).to_dict(orient="records")
    except Exception as e:
        error = str(e)

    return render_template(
        "data_loading.html",
        raw_summary=raw_summary,
        cleaned_summary=cleaned_summary,
        preview_cols=preview_cols,
        preview_rows=preview_rows,
        error=error,
    )


@app.route("/load")
def load_page():
    try:
        df = get_df()
        info = {
            "shape": df.shape,
            "head": df.head().to_html(classes="table data-table", index=False),
            "dtypes": df.dtypes.to_frame("dtype").to_html(classes="table data-table"),
        }
        error = None
    except Exception as e:
        info = None
        error = str(e)
    return render_template("load.html", info=info, error=error)


@app.route("/eda")
def eda_page():
    try:
        df = get_df()
        stats = get_summary_stats(df)
        chart_paths = generate_charts(df)
        chart_files = [os.path.basename(p) for p in chart_paths]
        summary = eda_module.summarize(df)
        images = chart_files
        error = None
    except Exception as e:
        stats = summary = images = None
        chart_files = []
        error = str(e)

    return render_template(
        "eda.html",
        stats=stats,
        chart_files=chart_files,
        summary=summary,
        images=images,
        error=error,
    )


# Alias
@app.route("/eda_page")
def eda():
    return eda_page()


@app.route("/feature-engg")
def feature_engg_page():
    try:
        df = get_df()
        normalized_df, standardized_df = fe_module.normalize(df, exclude_cols=TARGET_COLS)
        standardized_df = fe_module.outlier(standardized_df)
        _cache["standardized_df"] = standardized_df

        result = {
            "normalized_head": normalized_df.head().to_html(classes="table data-table", index=False),
            "standardized_head": standardized_df.head().to_html(classes="table data-table", index=False),
        }
        error = None
    except Exception as e:
        result = None
        error = str(e)
    return render_template("feature_engg.html", result=result, error=error)


# Alias
@app.route("/feature_engg")
def feature_engg():
    return feature_engg_page()


@app.route("/linear-regression")
def linear_regression_page():
    try:
        df = _cache.get("standardized_df", get_df())
        gd_result = lr_module.simple_linear_regression_GD(df)
        sk_result = lr_module.simple_linear_regression_sklearn(df)
        error = None
    except Exception as e:
        gd_result = sk_result = None
        error = str(e)
    return render_template(
        "linear_regression.html", gd_result=gd_result, sk_result=sk_result, error=error
    )


@app.route("/logistic-regression")
def logistic_regression_page():
    try:
        df = _cache.get("standardized_df", get_df())
        result = logit_module.simple_logistic_regression(df)
        error = None
    except Exception as e:
        result = None
        error = str(e)
    return render_template("logistic_regression.html", result=result, error=error)


@app.route("/train", methods=["GET", "POST"])
def train_model():
    error = None
    if request.method == "POST":
        try:
            from ml.trainer import train_model as run_train_model
            run_train_model()
        except Exception as e:
            error = f"Training failed: {e}"

    model_obj = load_trained_model()
    metrics = model_obj.get("metrics") if model_obj else None
    return render_template("train.html", metrics=metrics, error=error)


@app.route("/api/data-summary")
def api_data_summary():
    raw_df = load_data(DATA_PATH)
    return jsonify(get_data_summary(raw_df))


# =============================================================================
# CO1-CO6 & M1-M6 WORKBENCH, CLEANING & INSIGHTS ENDPOINTS (RideRush Port)
# =============================================================================

@app.route("/cleaning")
def cleaning_page():
    """
    Data Quality & Cleaning Page:
    Missing values, outlier treatment benchmarks, and retention statistics.
    """
    try:
        from placement_pipeline import load_pipeline_summary
        summary = load_pipeline_summary()
        return render_template("cleaning.html", summary=summary)
    except Exception as e:
        return render_template("cleaning.html", summary={}, error=str(e))


@app.route("/insights")
def insights_page():
    """
    Analytical Insights Page:
    Calculated empirical metrics and placement takeaways.
    """
    try:
        from placement_pipeline import load_pipeline_summary
        summary = load_pipeline_summary()
        return render_template("insights.html", summary=summary)
    except Exception as e:
        return render_template("insights.html", summary={}, error=str(e))


@app.route("/workbench", methods=["GET", "POST"])
def workbench_page():
    """
    Master ML Engineering & Model Workbench Page covering M1-M6 syllabus:
    Lifecycle, Regression Workbench, Classification, Tree Ensembles,
    Unsupervised Learning, Evaluation & Calibration, and ML Engineering.
    """
    from placement_pipeline import load_pipeline_summary
    from ml.regression_workbench import predict_workbench_salary_by_model
    from ml.m6_engineering import record_prediction_telemetry

    try:
        summary = load_pipeline_summary()
    except Exception as e:
        summary = {"error": str(e)}

    active_tab = request.args.get("tab", "regression")
    prediction_result = None
    form_values = {"CGPA": 8.2, "CodingTestScore": 82.5, "Internships": 2, "model_key": "polynomial"}

    if request.method == "POST":
        form_values = {
            "CGPA": float(request.form.get("CGPA", 8.2)),
            "CodingTestScore": float(request.form.get("CodingTestScore", 80.0)),
            "Internships": int(request.form.get("Internships", 1)),
            "model_key": request.form.get("model_key", "polynomial")
        }
        try:
            prediction_result = predict_workbench_salary_by_model(
                cgpa_val=form_values["CGPA"],
                coding_val=form_values["CodingTestScore"],
                internships_val=form_values["Internships"],
                model_key=form_values["model_key"]
            )
            record_prediction_telemetry(prediction_result["latency_ms"], form_values["model_key"])
        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template(
        "workbench.html",
        summary=summary,
        active_tab=active_tab,
        form_values=form_values,
        prediction_result=prediction_result
    )


@app.route("/api/workbench")
def api_workbench():
    """
    API endpoint returning regression workbench benchmarking data and metrics.
    """
    try:
        from placement_pipeline import load_pipeline_summary
        summary = load_pipeline_summary()
        return jsonify(summary.get("workbench", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workbench/summary")
def api_workbench_summary():
    """
    Comprehensive JSON API endpoint returning all M1-M6 model metrics and schemas.
    """
    try:
        from placement_pipeline import load_pipeline_summary
        summary = load_pipeline_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workbench/predict", methods=["GET", "POST"])
def api_workbench_predict():
    """
    JSON API endpoint for multi-model salary prediction with schema validation.
    Usage: GET /api/workbench/predict?cgpa=8.5&coding=80&internships=2&model=ridge
    """
    try:
        from ml.regression_workbench import predict_workbench_salary_by_model
        from ml.m6_engineering import record_prediction_telemetry

        if request.method == "POST" and request.is_json:
            data = request.get_json()
        else:
            data = request.args.to_dict() or (request.form.to_dict() if request.method == "POST" else {})

        cgpa = data.get("CGPA", data.get("cgpa", 8.0))
        coding = data.get("CodingTestScore", data.get("coding", 75.0))
        internships = data.get("Internships", data.get("internships", 1))
        model_key = data.get("model", data.get("model_key", "polynomial"))

        result = predict_workbench_salary_by_model(
            cgpa_val=cgpa,
            coding_val=coding,
            internships_val=internships,
            model_key=model_key
        )
        record_prediction_telemetry(result["latency_ms"], model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/workbench/batch-predict", methods=["POST"])
def api_workbench_batch_predict():
    """
    Batch prediction endpoint supporting JSON tabular records or raw CSV text.
    Max 1,000 records per request.
    """
    try:
        import pandas as pd
        from ml.m1_lifecycle import execute_batch_prediction

        model_key = request.args.get("model", "polynomial")
        if request.is_json:
            records = request.json.get("records", [])
            df_batch = pd.DataFrame(records)
            result = execute_batch_prediction(df_batch, model_key=model_key)
        else:
            csv_text = request.data.decode("utf-8")
            result = execute_batch_prediction(csv_text, model_key=model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/workbench/drift")
def api_workbench_drift():
    """
    Returns feature distribution drift audit and Kolmogorov-Smirnov test.
    """
    try:
        from placement_pipeline import load_pipeline_summary
        summary = load_pipeline_summary()
        drift = summary.get("m6_engineering", {}).get("feature_drift_analysis", {})
        return jsonify(drift)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/workbench/telemetry")
def api_workbench_telemetry():
    """
    Returns production serving telemetry (request counts, latency percentiles).
    """
    try:
        from ml.m6_engineering import get_monitoring_telemetry
        return jsonify(get_monitoring_telemetry())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Student Placement Web App on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
