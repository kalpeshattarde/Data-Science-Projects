"""Feature engineering for carbon trading decision models."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6
FUEL_FACTOR = {
    'Coal': 0.95,
    'Mixed Fuel': 0.75,
    'Natural Gas': 0.50,
    'Renewable': 0.25,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features aligned with the Streamlit app and notebooks."""
    out = df.copy()
    out['Emission_Gap'] = out['Emission_Produced_tCO2'] - out['Emission_Allowance_tCO2']
    out['Compliance_Pressure'] = out['Emission_Gap'] * out['Carbon_Price_USD_per_t']
    out['Cost_per_MWh'] = np.where(
        out['Energy_Demand_MWh'] > 0,
        out['Compliance_Cost_USD'] / (out['Energy_Demand_MWh'] + EPS),
        0,
    )
    out['Fuel_Carbon_Factor'] = out['Fuel_Type'].map(FUEL_FACTOR).fillna(0.50)
    return out
