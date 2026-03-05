"""Model training entrypoint."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingRegressor


def train_model(X, y):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def save_model(model, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
