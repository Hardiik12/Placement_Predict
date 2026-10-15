"""
models/predict.py
-----------------
Inference utilities for predicting student placement.
"""

from models.load import get_feature_info, predict_student

__all__ = ["predict_student", "get_feature_info"]
