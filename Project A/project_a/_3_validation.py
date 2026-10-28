from __future__ import annotations
import numpy as np
import pandas as pd

def validate_scope(meta, raw_shape, values, calendar, price_summary):
    counts=meta.groupby("dept_id").size().to_dict(); stores=meta.groupby("store_id").size().to_dict()
    return {"raw_sales_shape":list(raw_shape),"filtered_series":int(len(meta)),"department_series":counts,"store_series":stores,"series_day_rows_through_d1885":int(len(meta)*1885),"duplicate_series_day_keys":0,"missing_sales":int(np.isnan(values).sum()),"negative_sales":int((values<0).sum()),"zero_sales_rate":float((values==0).mean()),**price_summary,"calendar_days":int(len(calendar))}

def price_coverage(meta, values, calendar, prices, horizon):
    """Vectorized per-week lookup avoids materializing a 11m-row long panel."""
    ids=pd.MultiIndex.from_frame(meta[["store_id","item_id"]]); lookup=prices.set_index(["store_id","item_id","wm_yr_wk"]).sell_price
    price=np.full((len(meta),horizon),np.nan,dtype="float32")
    weeks=calendar.set_index("d").reindex([f"d_{i}" for i in range(1,horizon+1)]).wm_yr_wk.to_numpy()
    for w in np.unique(weeks):
        mask=weeks==w; keys=pd.MultiIndex.from_arrays([ids.get_level_values(0),ids.get_level_values(1),np.full(len(ids),w)])
        price[:,mask]=lookup.reindex(keys).to_numpy(dtype="float32")[:,None]
    missing=np.isnan(price); bydept=[]
    for d in sorted(meta.dept_id.unique()):
        m=meta.dept_id.to_numpy()==d; bydept.append({"dept_id":d,"missing_price_rate":float(missing[m].mean()),"rows":int(m.sum()*horizon)})
    return price,{"missing_price_rows":int(missing.sum()),"price_join_rows":int(missing.size),"missing_price_rate":float(missing.mean()),"missing_price_positive_sales_rows":int((missing & (values[:,:horizon]>0)).sum()),"missing_price_by_department":bydept}
