"""Data preprocessing utilities for silage sales data."""

from __future__ import annotations

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight cleaning aligned with notebook workflow."""
    cleaned = df.copy()
    cleaned.columns = [c.strip() for c in cleaned.columns]
    return cleaned
