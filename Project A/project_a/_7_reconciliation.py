from __future__ import annotations
import numpy as np
import pandas as pd

def reconcile_scenarios(store_forecast: pd.DataFrame, seed: int = 42, draws: int = 200) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Aggregate each joint scenario before quantiles, batched by target day to bound memory.

    The returned scenario frame is a deterministic audit sample (at most 200k rows); all
    requested DC quantiles are nevertheless calculated from every draw.
    """
    rng=np.random.default_rng(seed); qcols=[f"q{i:02d}" for i in range(5,100,5)]; qframes=[]; audit=[]
    for day,part in store_forecast.groupby("target_day",sort=True):
        keys=part[["item_id","dept_id"]].astype(str).agg("|".join,axis=1); codes,unique=pd.factorize(keys,sort=True); ng=len(unique); n=len(part)
        u=rng.random((n,draws)); grid=np.arange(.05,1,.05); vals=np.empty((n,draws),dtype="float32")
        source=part[qcols].to_numpy(dtype=float)
        for i in range(n): vals[i]=np.interp(u[i],grid,source[i],left=source[i,0],right=source[i,-1])
        aggregate=np.zeros((ng,draws),dtype="float32"); np.add.at(aggregate,codes,vals)
        keyframe=part[["item_id","dept_id"]].drop_duplicates().set_index(keys.drop_duplicates()).reindex(unique).reset_index(drop=True)
        qframe=keyframe.copy(); qframe["target_day"]=day
        for q in grid: qframe[f"q{int(q*100):02d}"]=np.quantile(aggregate,q,axis=1)
        qframes.append(qframe)
        remaining=max(0,200000-sum(len(x) for x in audit))
        if remaining:
            take=min(remaining,ng*draws); gi=np.repeat(np.arange(ng),draws)[:take]; di=np.tile(np.arange(draws),ng)[:take]
            audit.append(pd.DataFrame({"item_id":keyframe.item_id.to_numpy()[gi],"dept_id":keyframe.dept_id.to_numpy()[gi],"target_day":day,"scenario_draw":di,"dc_demand":aggregate[gi,di]}))
    return pd.concat(audit,ignore_index=True),pd.concat(qframes,ignore_index=True)
