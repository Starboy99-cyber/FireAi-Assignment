from __future__ import annotations
import numpy as np
import pandas as pd
from .forecasting import point_metrics, quantile_metrics

def fold_rows(meta, actual, forecasts, fold, model):
    point=point_metrics(actual,forecasts[.50]); rows=[{"fold":fold,"model":model,**point,"quantile":"point"}]
    rows += [{"fold":fold,"model":model,**x} for x in quantile_metrics(actual,forecasts)]
    for dept in sorted(meta.dept_id.unique()):
        ix=meta.dept_id.to_numpy()==dept; rows.append({"fold":fold,"model":model,"dept_id":dept,"quantile":"point",**point_metrics(actual[ix],forecasts[.50][ix])})
    for h in range(actual.shape[1]): rows.append({"fold":fold,"model":model,"day_ahead":h+1,"quantile":"point",**point_metrics(actual[:,h],forecasts[.50][:,h])})
    return rows

def summarize_policy(records):
    df=pd.DataFrame(records)
    if not df.empty:
        base=df[df.policy_name=="base_stock"].set_index(["period","seed"])
        df["pct_cost_change_vs_baseline"]=np.nan
        for i,r in df.iterrows():
            if r.policy_name!="base_stock" and (r.period,r.seed) in base.index:
                b=base.loc[(r.period,r.seed),"total_cost"]; df.loc[i,"pct_cost_change_vs_baseline"]=(r.total_cost-b)/b if b else np.nan
    return df
