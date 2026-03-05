"""Inference utility for saved silage price model."""

from __future__ import annotations

import joblib
import pandas as pd


def predict(model_path: str, input_df: pd.DataFrame):
    model = joblib.load(model_path)
    return model.predict(input_df)
