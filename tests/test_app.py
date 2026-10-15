"""
tests/test_app.py
------------------
Automated test suite for Student Placement Prediction Web App,
ML Engineering Workbench (M1–M6), Data Preprocessing, and Model APIs.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPL_DIR = os.path.join(BASE_DIR, ".cache", "matplotlib")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_DIR
import unittest
import json
import numpy as np
import pandas as pd
from app import app
from ml.m2_classification import sigmoid, softmax, compute_cross_entropy_loss
from ml.m3_trees import compute_tree_split_criteria, compute_cost_complexity_pruning_path
from ml.m5_evaluation import compute_comprehensive_regression_metrics, compute_mcnemar_test
from ml.m6_engineering import calculate_psi, detect_feature_drift, record_prediction_telemetry, get_monitoring_telemetry


class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student Placement Prediction", response.data)

    def test_data_loading_route(self):
        response = self.client.get("/data-loading")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data Loading", response.data)
        self.assertIn(b"Raw Dataset", response.data)

    def test_eda_route(self):
        response = self.client.get("/eda")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Exploratory Data Analysis", response.data)

    def test_feature_engg_route(self):
        response = self.client.get("/feature-engg")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Feature Engineering", response.data)

    def test_train_route_get(self):
        response = self.client.get("/train")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model Training", response.data)

    def test_predict_route_post(self):
        data = {
            "CGPA": "8.5",
            "AttendancePercent": "90.0",
            "CodingTestScore": "85.0",
            "AptitudeTestScore": "80.0",
            "SoftSkillsRating": "4.5",
            "MockInterviewScore": "80.0",
            "Internships": "2",
            "Projects": "3",
            "Workshops": "2",
            "Stream": "CS",
            "CollegeTier": "Tier 1",
            "HistoryOfBacklogs": "No",
        }
        response = self.client.post("/predict", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Result", response.data)

    def test_api_summary(self):
        response = self.client.get("/api/data-summary")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        json_data = response.get_json()
        self.assertIn("n_rows", json_data)
        self.assertEqual(json_data["n_rows"], 50000)

    # =========================================================================
    # WORKBENCH, CLEANING & INSIGHTS ROUTE TESTS
    # =========================================================================

    def test_cleaning_route(self):
        response = self.client.get("/cleaning")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Data Quality", response.data)

    def test_insights_route(self):
        response = self.client.get("/insights")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analytical Insights", response.data)

    def test_workbench_all_tabs(self):
        tabs = ["lifecycle", "regression", "classification", "trees", "unsupervised", "evaluation", "engineering", "syllabus"]
        for tab in tabs:
            response = self.client.get(f"/workbench?tab={tab}")
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"ML Engineering &amp; Model Workbench", response.data)

    def test_workbench_post_predict(self):
        form_data = {
            "CGPA": "8.4",
            "CodingTestScore": "88.0",
            "Internships": "2",
            "model_key": "polynomial"
        }
        response = self.client.post("/workbench?tab=regression", data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"LPA", response.data)

    def test_api_workbench(self):
        response = self.client.get("/api/workbench")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn("models", data)

    def test_api_workbench_summary(self):
        response = self.client.get("/api/workbench/summary")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn("m1_lifecycle", data)
        self.assertIn("m2_classification", data)
        self.assertIn("m3_trees", data)
        self.assertIn("m4_unsupervised", data)
        self.assertIn("m5_evaluation", data)
        self.assertIn("m6_engineering", data)

    def test_api_workbench_predict_get(self):
        response = self.client.get("/api/workbench/predict?cgpa=8.0&coding=75&internships=1&model=ridge")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn("predicted_salary_lpa", data)
        self.assertIn("latency_ms", data)
        self.assertGreater(data["predicted_salary_lpa"], 0)

    def test_api_workbench_batch_predict(self):
        csv_text = "CGPA,CodingTestScore,Internships\n8.5,85,2\n6.0,50,0\n"
        response = self.client.post(
            "/api/workbench/batch-predict?model=polynomial",
            data=csv_text,
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_processed"], 2)
        self.assertIn("preview", data)

    def test_api_workbench_drift(self):
        response = self.client.get("/api/workbench/drift")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn("psi_score", data)

    def test_api_workbench_telemetry(self):
        response = self.client.get("/api/workbench/telemetry")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIn("service_status", data)
        self.assertEqual(data["service_status"], "HEALTHY")

    # =========================================================================
    # ML MODULE UNIT TESTS
    # =========================================================================

    def test_m2_math_activations(self):
        z = np.array([-10.0, 0.0, 10.0])
        sig = sigmoid(z)
        self.assertAlmostEqual(sig[1], 0.5, places=4)
        self.assertLess(sig[0], 0.01)
        self.assertGreater(sig[2], 0.99)

        sm = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.sum(sm)), 1.0, places=4)

        loss = compute_cross_entropy_loss(np.array([1, 0]), np.array([0.9, 0.1]))
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 0.25)

    def test_m3_tree_split_criteria(self):
        crit = compute_tree_split_criteria([0.7, 0.3])
        self.assertEqual(crit["gini_impurity"], 0.42)
        self.assertGreater(crit["entropy"], 0.85)

    def test_m5_regression_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.5, 19.5, 30.2])
        metrics = compute_comprehensive_regression_metrics(y_true, y_pred)
        self.assertGreater(metrics["r2"], 0.95)
        self.assertLess(metrics["mae"], 1.0)

    def test_m6_drift_psi(self):
        base = np.random.normal(7.5, 1.0, 500)
        prod = np.random.normal(7.5, 1.0, 500)
        drift = detect_feature_drift(base, prod)
        self.assertIn(drift["severity"], ["NEGLIGIBLE", "MODERATE"])


if __name__ == "__main__":
    unittest.main()
