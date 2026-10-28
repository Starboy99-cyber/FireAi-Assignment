from __future__ import annotations
import numpy as np
import pandas as pd

def _runs(row):
    z = row == 0; padded = np.r_[False,z,False]; change=np.flatnonzero(padded[1:] != padded[:-1]); lengths=change[1::2]-change[::2]
    return lengths

def diagnostics(meta: pd.DataFrame, sales: np.ndarray, cutoff: int) -> pd.DataFrame:
    arr=sales[:,:cutoff]; rows=[]
    for i,r in meta.reset_index(drop=True).iterrows():
        x=arr[i]; runs=_runs(x); trailing=int(np.argmax(x[::-1] != 0)) if (x[::-1] != 0).any() else cutoff
        rows.append({"id":r.id,"item_id":r.item_id,"store_id":r.store_id,"dept_id":r.dept_id,"zero_sales_rate":float((x==0).mean()),"active_days":int((x>0).sum()),"recent_active_28":int((x[-28:]>0).sum()),"recent_active_56":int((x[-56:]>0).sum()),"recent_active_90":int((x[-90:]>0).sum()),"longest_zero_run":int(runs.max()) if len(runs) else 0,"zero_runs_ge_7":int((runs>=7).sum()),"zero_runs_ge_14":int((runs>=14).sum()),"trailing_zero_days":trailing,"possible_censoring_flag":bool(trailing==0 and (x[-28:]>0).sum()>10)})
    return pd.DataFrame(rows)

def scenarios(diag: pd.DataFrame) -> pd.DataFrame:
    out=diag[["id","dept_id","possible_censoring_flag","recent_active_28","zero_sales_rate"]].copy()
    out["observed_sales_multiplier"] = 1.0
    out["conservative_recovery_multiplier"] = np.where(out.possible_censoring_flag,1.05,1.0)
    out["higher_demand_multiplier"] = np.where(out.possible_censoring_flag,1.15,1.0)
    out["evidence"] = np.where(out.possible_censoring_flag,"Recent activity with an isolated zero pattern; sensitivity only","No evidence sufficient to adjust observed sales")
    return out
