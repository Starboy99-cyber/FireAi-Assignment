from __future__ import annotations
import numpy as np
import pandas as pd
from .config import ASSUMPTIONS
from .policy import policy, lognormal_parameters, round_lead_time, allocate_capacity

def simulate(meta: pd.DataFrame, demand: np.ndarray, forecast_q: np.ndarray, prices: np.ndarray, mean_demand: np.ndarray, seed: int, proposed: bool, disruption: str="normal", warning: bool=False):
    """Event order: arrivals, sell/lost sales, holding, transfer decision, Monday external PO."""
    n,T=demand.shape; rng=np.random.default_rng(seed); dept=meta.dept_id.to_numpy(); lead=np.array([ASSUMPTIONS['supplier_lead_time_days'][d] for d in dept],float)
    cap=mean_demand*ASSUMPTIONS['store_inventory_cap_days']; store=np.minimum(cap,mean_demand*ASSUMPTIONS['initial_stock_days']); transit=[np.zeros(n) for _ in range(2)]; dc=float((mean_demand*14).sum()); dc_orders=[]
    outbound_cap=ASSUMPTIONS['dc_capacity_multiplier']*mean_demand.sum(); costs={"lost_sales_cost":0.,"holding_cost":0.,"purchase_order_cost":0.,"transfer_cost":0.}; totals={"demand_units":0.,"fulfilled_units":0.,"lost_units":0.,"transfer_units":0.,"external_units":0.}; traces=[]
    for t in range(T):
        store+=transit.pop(0); transit.append(np.zeros(n))
        arrivals=[o for o in dc_orders if o[0]==t]; dc+=sum(o[1] for o in arrivals); dc_orders=[o for o in dc_orders if o[0]!=t]
        dem=demand[:,t]; fulfilled=np.minimum(store,dem); lost=dem-fulfilled; store-=fulfilled
        p=np.where(np.isfinite(prices[:,t]),prices[:,t],np.nanmedian(prices[np.isfinite(prices)]) if np.isfinite(prices).any() else 1.)
        costs['lost_sales_cost']+=float((lost*p*ASSUMPTIONS['lost_sale_cost_rate']).sum()); costs['holding_cost']+=float((store*p*ASSUMPTIONS['holding_cost_rate_annual']/365).sum())
        totals['demand_units']+=dem.sum();totals['fulfilled_units']+=fulfilled.sum();totals['lost_units']+=lost.sum()
        in_transit=sum(transit); state={"forecast_q":forecast_q[:,t],"mean_demand":mean_demand,"lead_days":lead,"store_cap":cap,"store_on_hand":store,"store_in_transit":in_transit,"dc_on_hand":np.full(n,dc/n),"dc_on_order":np.zeros(n)}
        decision=policy(state,proposed); transfer=allocate_capacity(decision['transfer_request'],store,forecast_q[:,t],outbound_cap); transfer=np.minimum(transfer,dc); dc-=transfer.sum(); transit[-1]+=transfer; totals['transfer_units']+=transfer.sum(); costs['transfer_cost']+=float(transfer.sum()*ASSUMPTIONS['transfer_cost_per_unit'])
        # Monday in M5 d_1 is Saturday, hence day index t with weekday ((t+5)%7)==0 is Monday.
        if ((t+5)%7)==0:
            for d in np.unique(dept):
                idx=dept==d; qty=float(decision['external_request']*idx.mean())
                if warning and t<14 and d=='FOODS_3': qty*=1.25
                if qty>0:
                    m=lead[idx][0]*(2 if disruption!='normal' and d=='FOODS_3' and t<21 else 1); mu,sigma=lognormal_parameters(m,ASSUMPTIONS['lead_time_cv']); arrival=t+int(round_lead_time(rng.lognormal(mu,sigma), 'ceil'))
                    dc_orders.append((arrival,qty)); totals['external_units']+=qty; costs['purchase_order_cost']+=ASSUMPTIONS['purchase_order_cost_per_department']
        traces.append({"day":t+1,"dc_on_hand":dc,"store_on_hand":float(store.sum()),"demand":float(dem.sum()),"fulfilled":float(fulfilled.sum()),"lost":float(lost.sum()),"transfer":float(transfer.sum()),"outbound_capacity":outbound_cap,"capacity_observed":float(transfer.sum())})
    total=sum(costs.values()); fill=totals['fulfilled_units']/totals['demand_units'] if totals['demand_units'] else np.nan
    return {**costs,**totals,"total_cost":total,"fill_rate":fill,"fill_rate_shortfall":max(0,ASSUMPTIONS['store_fill_rate_target']-fill),"avg_store_inventory":float(np.mean([x['store_on_hand'] for x in traces]))},pd.DataFrame(traces)
