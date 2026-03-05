"""Prediction utility for saved IEX model pipeline."""

from __future__ import annotations

import joblib
import pandas as pd


def predict(model_path: str, features: pd.DataFrame):
    model = joblib.load(model_path)
    return model.predict(features)
