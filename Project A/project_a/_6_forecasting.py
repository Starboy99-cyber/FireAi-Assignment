from __future__ import annotations
import numpy as np
import pandas as pd

QUANTILES = np.round(np.arange(.05, 1.0, .05),2)

def _weekday_samples(history: np.ndarray, target_days: np.ndarray, origin: int, days_back: int = 364) -> list[np.ndarray]:
    start=max(0, origin-days_back); idx=np.arange(start,origin); result=[]
    for t in target_days:
        comparable=idx[((idx+1)%7)==((t+1)%7)]
        result.append(history[:,comparable])
    return result

def seasonal_quantile_forecast(history: np.ndarray, origin: int, horizon: int, model="seasonal_empirical") -> dict[float,np.ndarray]:
    """Comparable weekday empirical distribution, using only history before origin."""
    targets=np.arange(origin, origin+horizon)
    samples=_weekday_samples(history,targets,origin)
    return {q:np.stack([np.quantile(s,q,axis=1) for s in samples],axis=1).astype("float32") for q in QUANTILES}

def moving_average_forecast(history: np.ndarray, origin: int, horizon: int) -> dict[float,np.ndarray]:
    h=history[:,max(0,origin-56):origin]; mean=h.mean(axis=1); positive=h[h>0]
    # distribution uses an item-series empirical residual scale, appropriate for zero-heavy demand.
    sd=h.std(axis=1); z={.05:-1.645,.10:-1.282,.15:-1.036,.20:-.842,.25:-.674,.30:-.524,.35:-.385,.40:-.253,.45:-.126,.50:0,.55:.126,.60:.253,.65:.385,.70:.524,.75:.674,.80:.842,.85:1.036,.90:1.282,.95:1.645}
    return {q:np.maximum(0,mean[:,None]+z[float(q)]*sd[:,None]).repeat(horizon,axis=1).astype("float32") for q in QUANTILES}

def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err=np.abs(actual-pred); denom=float(actual.sum())
    return {"mae":float(err.mean()),"wape":float(err.sum()/denom) if denom else np.nan}

def quantile_metrics(actual: np.ndarray, forecasts: dict[float,np.ndarray]) -> list[dict]:
    rows=[]
    for q,p in forecasts.items():
        e=actual-p; pin=np.maximum(q*e,(q-1)*e).mean(); rows.append({"quantile":q,"pinball_loss":float(pin),"coverage":float((actual<=p).mean())})
    rows.append({"quantile":"interval_05_95","pinball_loss":np.nan,"coverage":np.nan,"interval_width":float((forecasts[.95]-forecasts[.05]).mean())})
    return rows

def to_frame(meta: pd.DataFrame, forecast: dict[float,np.ndarray], origin: int, scenario="observed_sales", model="seasonal_empirical") -> pd.DataFrame:
    n,h=next(iter(forecast.values())).shape; base=meta[["id","item_id","store_id","dept_id"]].reset_index(drop=True)
    out=base.loc[base.index.repeat(h)].reset_index(drop=True)
    out["forecast_origin"] = f"d_{origin}"; out["target_day"] = np.tile(np.arange(origin+1,origin+h+1),n); out["day_ahead"] = np.tile(np.arange(1,h+1),n); out["model"]=model; out["scenario"]=scenario
    for q,a in forecast.items(): out[f"q{int(round(q*100)):02d}"]=a.reshape(-1)
    return out

def validate_forecasts(frame: pd.DataFrame):
    cols=[f"q{i:02d}" for i in range(5,100,5)]
    if set(cols)-set(frame): raise ValueError("Missing required forecast quantiles")
    x=frame[cols].to_numpy();
    if (x < 0).any() or (np.diff(x,axis=1)<-1e-6).any(): raise ValueError("Forecasts must be nonnegative and quantile ordered")
