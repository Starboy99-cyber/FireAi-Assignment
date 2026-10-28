from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLS = ["lag_1","lag_7","lag_14","lag_28","lag_56","roll_mean_7","roll_mean_28","roll_mean_56","roll_std_7","roll_std_28","roll_std_56","weekday_num","month_num","year_num","is_weekend","snap_CA","sell_price","price_missing"]
CATEGORICAL_COLS = ["item_id","store_id","dept_id","event_name_1","event_type_1","event_name_2","event_type_2"]

def build_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features at t are only functions of observations before t."""
    x = panel.sort_values(["id","day_num"]).copy()
    g = x.groupby("id", sort=False)["sales"]
    for lag in (1,7,14,28,56): x[f"lag_{lag}"] = g.shift(lag)
    for win in (7,28,56):
        shifted = g.shift(1)
        x[f"roll_mean_{win}"] = shifted.groupby(x["id"], sort=False).transform(lambda s: s.rolling(win, min_periods=1).mean())
        x[f"roll_std_{win}"] = shifted.groupby(x["id"], sort=False).transform(lambda s: s.rolling(win, min_periods=2).std()).fillna(0)
    return x

def assert_no_future_values(panel: pd.DataFrame, cutoff: int) -> None:
    if (panel.day_num > cutoff).any(): raise ValueError("Feature frame includes observations after forecast cutoff")
