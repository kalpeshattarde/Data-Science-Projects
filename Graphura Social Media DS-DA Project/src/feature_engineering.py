"""Feature engineering for content and influencer scoring."""
from __future__ import annotations
import pandas as pd

def build_engagement_rate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {'Likes','Comments','Shares','Reach'}.issubset(set(out.columns)):
        out['Engagement_Rate'] = (out['Likes'] + out['Comments'] + out['Shares']) / out['Reach'].replace(0, 1)
    return out
