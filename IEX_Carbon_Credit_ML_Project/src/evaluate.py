"""Evaluation metrics for classification models."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate_classification(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'report': classification_report(y_true, y_pred),
    }
