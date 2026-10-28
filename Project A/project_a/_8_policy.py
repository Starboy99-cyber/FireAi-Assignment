from __future__ import annotations
import numpy as np
try:
    from .config import ASSUMPTIONS
except ImportError:  # Support running this module directly from the project directory.
    from config import ASSUMPTIONS

def lognormal_parameters(mean: float, cv: float):
    s2=np.log(1+cv**2); return np.log(mean)-s2/2,np.sqrt(s2)

def round_lead_time(samples, rule="ceil"):
    x=np.asarray(samples); return np.maximum(1, np.ceil(x) if rule=="ceil" else np.rint(x)).astype(int)

def policy(state: dict, proposed: bool = True) -> dict:
    """Order-up-to policy. State uses item-store vector quantities and demand-rate units/day."""
    demand=state["forecast_q"] if proposed else state["mean_demand"]
    protection=state["lead_days"]+ASSUMPTIONS["dc_to_store_transit_days"]+1
    target=np.minimum(state["store_cap"], demand*protection)
    transfer=np.maximum(0,target-(state["store_on_hand"]+state["store_in_transit"])); transfer=np.minimum(transfer,state["store_cap"]-state["store_on_hand"])
    dc_target=demand*(state["lead_days"]+7); external=np.maximum(0,dc_target-(state["dc_on_hand"]+state["dc_on_order"])).sum()
    return {"transfer_request":transfer,"external_request":external}

def allocate_capacity(request: np.ndarray, stock: np.ndarray, demand: np.ndarray, capacity: float) -> np.ndarray:
    """Allocate scarce DC capacity by shortage urgency; never exceed requests, stock, or daily capacity."""
    available=min(float(stock.sum()),float(capacity)); urgency=np.divide(demand,np.maximum(stock,1e-6)); order=np.argsort(-urgency); out=np.zeros_like(request,float)
    for i in order:
        give=min(request[i],available); out[i]=give; available-=give
        if available<=1e-9: break
    return out
