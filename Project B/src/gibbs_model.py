"""Compile-free Gibbs sampler for the hierarchical geo MMM.

The sampler targets a Gaussian hierarchical model exactly conditional on the
fixed, causal media transformations. It is used when a native PyMC backend is
unavailable and is deliberately explicit about that conditioning.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from .validation import validate_data, chronological_split
from .transformations import add_media_features

FEATURES=[f"media_{i}" for i in range(5)]+["Organic_channel0_impression","Promo","competitor_sales_control","sentiment_score_control","sin1","cos1","sin2","cos2"]

def _design(d: pd.DataFrame, origin: pd.Timestamp, means=None, sds=None):
    x=d.copy(); e=(x.time-origin).dt.days.to_numpy()/7
    for k in (1,2): x[f"sin{k}"]=np.sin(2*np.pi*k*e/52.18); x[f"cos{k}"]=np.cos(2*np.pi*k*e/52.18)
    raw=x[FEATURES].to_numpy(float)
    if means is None:
        means=raw.mean(0); sds=raw.std(0); sds=np.where(sds==0,1,sds)
    z=(raw-means)/sds
    return np.column_stack([np.ones(len(x)),z]),means,sds

def _rhat(chains: np.ndarray) -> float:
    # chains has shape chains × draws. Standard split-chain R-hat.
    m,n=chains.shape; means=chains.mean(1); w=chains.var(1,ddof=1).mean(); b=n*means.var(ddof=1)
    return float(np.sqrt(((n-1)/n*w+b/n)/max(w,1e-12)))

def _ess_lag1(chains: np.ndarray) -> float:
    r=[]
    for x in chains:
        if np.std(x)==0: r.append(0.)
        else: r.append(np.corrcoef(x[:-1],x[1:])[0,1])
    rho=float(np.clip(np.nanmean(r),-0.99,.99)); return float(chains.size*(1-rho)/(1+rho))

def _gibbs(X,y,geo,n_geo,draws,tune,seed):
    """Two-chain Gibbs sampler for y=Xb+u_geo+epsilon, u~N(0,tau²)."""
    n,p=X.shape; xtx=X.T@X; chains_b=[]; chains_u=[]; chains_s=[]; chains_t=[]
    for chain in range(2):
        rng=np.random.default_rng(seed+1009*chain); b=np.zeros(p); b[0]=y.mean(); u=np.zeros(n_geo); s2=np.var(y); tau2=np.var(y)/4
        keep_b=[]; keep_u=[]; keep_s=[]; keep_t=[]
        prior_prec=np.full(p,1/9); prior_prec[0]=1/100
        for it in range(tune+draws):
            resid=y-u[geo]
            precision=xtx/s2+np.diag(prior_prec); cov=np.linalg.inv(precision); mean=cov@(X.T@resid/s2)
            b=mean+np.linalg.cholesky(cov)@rng.normal(size=p)
            base=y-X@b
            counts=np.bincount(geo,minlength=n_geo); sums=np.bincount(geo,weights=base,minlength=n_geo)
            var=1/(counts/s2+1/tau2); u=var*(sums/s2)+np.sqrt(var)*rng.normal(size=n_geo)
            # Identify the intercept separately from geo deviations. This
            # sum-to-zero reparameterization materially improves Gibbs mixing.
            shift=u.mean(); u=u-shift; b[0]=b[0]+shift
            err=y-X@b-u[geo]; s2=1/rng.gamma(2+n/2,1/(.01+err@err/2))
            tau2=1/rng.gamma(2+n_geo/2,1/(.01+u@u/2))
            if it>=tune: keep_b.append(b.copy()); keep_u.append(u.copy()); keep_s.append(s2); keep_t.append(tau2)
        chains_b.append(np.array(keep_b)); chains_u.append(np.array(keep_u)); chains_s.append(np.array(keep_s)); chains_t.append(np.array(keep_t))
    return np.array(chains_b),np.array(chains_u),np.array(chains_s),np.array(chains_t)

def run_gibbs(data_path: str, output_dir: str, holdout_start: str|None, seed: int, draws: int=500, tune: int=500):
    out=Path(output_dir); tables=out/"tables"; reports=out/"reports"; figures=out/"figures"; model=out/"model"
    for p in [tables,reports,figures,model]: p.mkdir(parents=True,exist_ok=True)
    os.environ["MPLCONFIGDIR"]=str(out/".matplotlib")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw=pd.read_csv(data_path); d,facts=validate_data(raw); d,_=add_media_features(d,.5)
    train,hold,actual=chronological_split(d,holdout_start); train=train.sort_values(["time","geo"]); hold=hold.sort_values(["time","geo"])
    X,means,sds=_design(train,d.time.min()); Xh,_,_=_design(hold,d.time.min(),means,sds)
    y=(train.conversions*train.revenue_per_conversion).to_numpy()/100_000; yh=(hold.conversions*hold.revenue_per_conversion).to_numpy()
    geos=np.array(sorted(d.geo.unique())); gi=pd.Categorical(train.geo,categories=geos).codes; gh=pd.Categorical(hold.geo,categories=geos).codes
    b,u,s2,tau2=_gibbs(X,y,gi,len(geos),draws,tune,seed)
    flat_b=b.reshape(-1,b.shape[-1]); flat_u=u.reshape(-1,u.shape[-1]); flat_s=s2.reshape(-1); rng=np.random.default_rng(seed+7)
    pred=(Xh@flat_b.T+flat_u[:,gh].T+rng.normal(size=(len(hold),len(flat_s)))*np.sqrt(flat_s))*100_000
    mean=pred.mean(1); lo,hi=np.quantile(pred,[.05,.95],axis=1)
    metrics={"RMSE":float(np.sqrt(np.mean((yh-mean)**2))),"MAE":float(np.mean(abs(yh-mean))),"WAPE":float(np.sum(abs(yh-mean))/np.sum(abs(yh))),"interval_90_coverage":float(np.mean((yh>=lo)&(yh<=hi))),"interval_90_mean_width":float(np.mean(hi-lo))}
    prediction=hold[["geo","time"]].copy(); prediction["revenue"]=yh; prediction["prediction_mean"]=mean; prediction["prediction_p05"]=lo; prediction["prediction_p95"]=hi; prediction.to_csv(tables/"mcmc_holdout_predictions.csv",index=False)
    names=["intercept"]+FEATURES; rows=[]
    for j,name in enumerate(names):
        c=b[:,:,j]; rows.append({"parameter":name,"mean":float(c.mean()),"p05":float(np.quantile(c,.05)),"p95":float(np.quantile(c,.95)),"r_hat":_rhat(c),"ess_lag1_approx":_ess_lag1(c)})
    for name,c in [("sigma",np.sqrt(s2)),("sigma_geo",np.sqrt(tau2))]: rows.append({"parameter":name,"mean":float(c.mean()),"p05":float(np.quantile(c,.05)),"p95":float(np.quantile(c,.95)),"r_hat":_rhat(c),"ess_lag1_approx":_ess_lag1(c)})
    summary=pd.DataFrame(rows); summary.to_csv(tables/"mcmc_posterior_summary.csv",index=False)
    # Posterior predictive diagnostic from 200 random train draws.
    ind=rng.choice(len(flat_s),size=min(200,len(flat_s)),replace=False); ppc=(X@flat_b[ind].T+rng.normal(size=(len(train),len(ind)))*np.sqrt(flat_s[ind]))
    fig,ax=plt.subplots(figsize=(8,4)); ax.hist(y,bins=45,density=True,alpha=.6,label="observed training revenue (100k)"); ax.hist(ppc.ravel(),bins=45,density=True,alpha=.35,label="posterior predictive"); ax.legend(); ax.set_title("Posterior-predictive check: Gibbs hierarchical MMM"); fig.tight_layout(); fig.savefig(figures/"mcmc_posterior_predictive.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(9,8),sharex=True)
    for i,ax in enumerate(axes): ax.plot(b[0,:,i+1],alpha=.75,label="chain 1"); ax.plot(b[1,:,i+1],alpha=.75,label="chain 2"); ax.set_ylabel(f"Channel{i}")
    axes[0].legend(); axes[-1].set_xlabel("post-warmup draw"); fig.suptitle("Gibbs traces: sampled media coefficients"); fig.tight_layout(); fig.savefig(figures/"mcmc_trace_media_coefficients.png",dpi=150); plt.close(fig)
    group=prediction.groupby("time")[["revenue","prediction_mean","prediction_p05","prediction_p95"]].sum(); fig,ax=plt.subplots(figsize=(10,4)); ax.plot(group.index,group.revenue,label="actual"); ax.plot(group.index,group.prediction_mean,label="posterior mean"); ax.fill_between(group.index,group.prediction_p05,group.prediction_p95,alpha=.25,label="90% interval"); ax.legend(); ax.set_title("Holdout national revenue: hierarchical Gibbs MMM"); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures/"mcmc_holdout_forecast.png",dpi=150); plt.close(fig)
    np.savez_compressed(model/"gibbs_posterior_chains.npz",beta=b,geo_effect=u,sigma2=s2,tau2=tau2)
    report={"data_quality":facts,"split":{"actual_holdout_start":str(actual.date()),"train_rows":len(train),"holdout_rows":len(hold),"train_weeks":int(train.time.nunique()),"holdout_weeks":int(hold.time.nunique())},"sampling":{"method":"custom Gibbs","chains":2,"draws_per_chain":draws,"tune_per_chain":tune,"max_rhat":float(summary.r_hat.max()),"min_ess_lag1_approx":float(summary.ess_lag1_approx.min())},"holdout_metrics":metrics,"model":"Gaussian hierarchical geo-intercept MMM with sampled channel/control/seasonal coefficients","transformation_limitation":"Geometric adstock alpha=0.5 and median Hill saturation are fixed; a compiler-enabled PyMC implementation is required to jointly estimate them."}
    (reports/"mcmc_run_report.json").write_text(json.dumps(report,indent=2)); return report
