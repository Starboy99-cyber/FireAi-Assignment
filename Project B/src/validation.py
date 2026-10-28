"""Schema and data-quality checks with no silent correction."""
from __future__ import annotations
import pandas as pd

REQUIRED = {"geo", "time", "conversions", "revenue_per_conversion", "population", "Promo",
            "Organic_channel0_impression", "competitor_sales_control", "sentiment_score_control"}
REQUIRED |= {f"Channel{i}_{kind}" for i in range(5) for kind in ("impression", "spend")}

def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    d = df.copy(); d["time"] = pd.to_datetime(d["time"], errors="raise")
    duplicated = int(d.duplicated(["geo", "time"]).sum())
    weekly = d.groupby("time")["geo"].nunique().sort_index()
    value_columns = [c for c in d.columns if c.endswith(("_spend", "_impression"))] + ["conversions", "population"]
    negatives = {c: int((d[c] < 0).sum()) for c in value_columns}
    facts = {"rows": len(d), "geographies": int(d.geo.nunique()), "weeks": int(d.time.nunique()),
             "duplicates": duplicated, "date_start": str(d.time.min().date()), "date_end": str(d.time.max().date()),
             "negative_values": negatives, "nonpositive_rpc": int((d.revenue_per_conversion <= 0).sum()),
             "zero_spend": {f"Channel{i}": int((d[f"Channel{i}_spend"] == 0).sum()) for i in range(5)}}
    if duplicated: raise ValueError(f"Duplicate geo-week rows: {duplicated}")
    if any(negatives.values()) or facts["nonpositive_rpc"]: raise ValueError("Invalid negative or nonpositive economic values")
    return d, facts

def chronological_split(df: pd.DataFrame, holdout_start: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    weeks = pd.DatetimeIndex(sorted(df.time.unique()))
    requested = pd.Timestamp(holdout_start) if holdout_start else weeks[-13]
    actual = weeks[weeks >= requested][0] if (weeks >= requested).any() else None
    if actual is None: raise ValueError("holdout start is after data coverage")
    train, holdout = df[df.time < actual].copy(), df[df.time >= actual].copy()
    if train.time.nunique() < 52 or not len(holdout): raise ValueError("Split needs at least 52 training weeks and nonempty holdout")
    return train, holdout, actual
