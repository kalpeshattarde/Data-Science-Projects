"""Evaluation helpers for regression outputs."""
from __future__ import annotations
from sklearn.metrics import r2_score, mean_squared_error

def evaluate(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred), 'RMSE': mean_squared_error(y_true, y_pred, squared=False)}
