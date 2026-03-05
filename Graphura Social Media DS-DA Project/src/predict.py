"""Prediction helper for saved model."""
from __future__ import annotations
import joblib

def predict(model_path, X):
    model = joblib.load(model_path)
    return model.predict(X)
