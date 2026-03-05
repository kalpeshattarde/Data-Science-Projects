"""Training helpers for trade-action classification."""

from __future__ import annotations

from pathlib import Path

import joblib
from xgboost import XGBClassifier


def train_model(X, y):
    """Train a baseline XGBoost classifier."""
    model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='mlogloss',
    )
    model.fit(X, y)
    return model


def save_model(model, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
