"""
tests/test_app.py
------------------
Automated test suite for Flask routes, dataset loading, and ML model inference.
"""

import unittest
from app import app


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


if __name__ == "__main__":
    unittest.main()
