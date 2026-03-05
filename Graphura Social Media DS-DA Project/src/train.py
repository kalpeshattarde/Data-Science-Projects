"""Training entrypoint for Graphura model."""
from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
