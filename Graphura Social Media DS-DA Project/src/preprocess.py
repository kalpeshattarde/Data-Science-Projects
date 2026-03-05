"""Preprocessing utilities for Graphura social media data."""
from __future__ import annotations
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().replace(' ', '_') for c in out.columns]
    return out
