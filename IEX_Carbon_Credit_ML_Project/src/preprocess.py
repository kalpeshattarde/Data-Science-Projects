"""Data preprocessing utilities for IEX carbon trading data."""

from __future__ import annotations

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    return pd.read_csv(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and trim string fields."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for col in out.select_dtypes(include='object').columns:
        out[col] = out[col].astype(str).str.strip()
    return out
