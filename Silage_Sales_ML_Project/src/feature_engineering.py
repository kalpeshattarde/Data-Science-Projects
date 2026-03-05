"""Feature engineering helpers for silage pricing."""

from __future__ import annotations

import pandas as pd


def add_quality_band(df: pd.DataFrame) -> pd.DataFrame:
    """Create a categorical quality feature from moisture percent."""
    out = df.copy()
    m = out.get('Moisture_Content_%')
    if m is not None:
        out['Silage_Quality'] = m.apply(lambda x: 'High Quality Silage' if 62 <= float(x) <= 68 else 'Average Quality Silage')
    return out


def add_discount_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Create discount rate feature from quantity slabs."""
    out = df.copy()
    q = out.get('Quantity_MT')
    if q is not None:
        def slab(v: float) -> float:
            if v < 5:
                return 0.025
            if v < 10:
                return 0.05
            if v < 25:
                return 0.10
            return 0.15
        out['Discount_rate'] = q.astype(float).apply(slab)
    return out
