from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

SUPPORTED = (".csv", ".parquet")

def resolve_input(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file(): return path.resolve()
    for suffix in SUPPORTED:
        candidate = path.with_suffix(suffix)
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve input '{path}'. Checked exact path and: {', '.join(str(path.with_suffix(s)) for s in SUPPORTED)}")

def read_input(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)

def discover_paths(sales_path, prices_path, calendar_path) -> dict[str, Path]:
    return {"sales": resolve_input(sales_path), "prices": resolve_input(prices_path), "calendar": resolve_input(calendar_path)}

def load_scope(paths: dict[str, Path], stores: tuple[str, ...], departments: tuple[str, ...], max_series: int | None = None):
    sales = read_input(paths["sales"])
    day_cols = [c for c in sales.columns if c.startswith("d_") and c[2:].isdigit()]
    if not day_cols: raise ValueError("Sales input has no d_<integer> columns")
    scoped = sales.loc[sales.store_id.isin(stores) & sales.dept_id.isin(departments)].copy()
    if max_series: scoped = scoped.iloc[:max_series].copy()
    if scoped["id"].duplicated().any(): raise ValueError("Duplicate series IDs after filtering")
    values = scoped[day_cols].to_numpy(dtype="float32", copy=True)
    if pd.isna(values).any(): raise ValueError("Missing sales values found; refusing to coerce them to zero")
    if (values < 0).any(): raise ValueError("Negative sales values found")
    calendar, prices = read_input(paths["calendar"]), read_input(paths["prices"])
    if calendar.d.duplicated().any(): raise ValueError("Duplicate calendar daily keys")
    if prices.duplicated(["store_id","item_id","wm_yr_wk"]).any(): raise ValueError("Duplicate store-item-week price keys")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    return scoped, day_cols, values, calendar, prices, sales.shape

def input_metadata(paths: dict[str, Path], frames: dict[str, pd.DataFrame]) -> dict:
    return {k: {"path": str(paths[k]), "bytes": paths[k].stat().st_size, "rows": len(v), "columns": list(v.columns)} for k,v in frames.items()}

def price_matrix(meta: pd.DataFrame, day_cols: list[str], calendar: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    days = pd.DataFrame({"d": day_cols, "day_num": [int(d[2:]) for d in day_cols]}).merge(calendar, on="d", how="left", validate="one_to_one")
    if days.date.isna().any(): raise ValueError("Unmatched calendar days")
    left = meta[["id","item_id","store_id","dept_id"]].merge(days[["d","day_num","wm_yr_wk"]], how="cross")
    joined = left.merge(prices, on=["store_id","item_id","wm_yr_wk"], how="left", validate="many_to_one")
    joined["price_missing"] = joined.sell_price.isna()
    stats = {"missing_price_rows": int(joined.price_missing.sum()), "price_join_rows": len(joined), "missing_price_rate": float(joined.price_missing.mean())}
    return joined, stats
