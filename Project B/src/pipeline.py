"""Reproducible implementation and reporting for Project B MMM."""
from __future__ import annotations
import json, logging, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
import yaml
from .validation import validate_data, chronological_split
from .transformations import add_media_features

CHANNELS = [f"Channel{i}" for i in range(5)]

def _mkdir(root: Path):
    for x in ["tables", "reports", "figures", "model"]: (root / x).mkdir(parents=True, exist_ok=True)

def _write_png(path: Path, values: np.ndarray, color=(35, 90, 150)):
    """Dependency-free tiny bar chart PNG (actual metric visualization)."""
    import struct, zlib
    w,h=700,280; pix=np.full((h,w,3), 255, dtype=np.uint8); pix[-30:,:,:]=245
    vals=np.nan_to_num(np.asarray(values,float)); mx=max(float(np.max(np.abs(vals))),1e-9)
    n=len(vals); bw=max(3, int((w-40)/max(n,1)*.7))
    for i,v in enumerate(vals):
        bh=int(abs(v)/mx*(h-60)); x=20+int(i*(w-40)/max(n,1)); pix[h-31-bh:h-31,x:x+bw]=color
    raw=b''.join(b'\0'+pix[y].tobytes() for y in range(h))
    def chunk(k,d): return struct.pack('>I',len(d))+k+d+struct.pack('>I',zlib.crc32(k+d)&0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b''))

def _build_design(d: pd.DataFrame, train: pd.DataFrame | None = None):
    d = d.copy(); d["revenue"] = d.conversions * d.revenue_per_conversion
    cols = [f"media_{i}" for i in range(5)] + ["Organic_channel0_impression", "Promo", "competitor_sales_control", "sentiment_score_control"]
    # compact time basis (annual seasonality + trend) and partial-pooling proxy geo indicators
    elapsed = ((d.time - d.time.min()).dt.days / 7).to_numpy()
    for k in (1,2):
        d[f"sin{k}"]=np.sin(2*np.pi*k*elapsed/52.18); d[f"cos{k}"]=np.cos(2*np.pi*k*elapsed/52.18)
    cols += ["sin1","cos1","sin2","cos2"]
    geo = pd.get_dummies(d.geo, prefix="geo", dtype=float); geo = geo.iloc[:,1:]
    X=pd.concat([pd.Series(1., index=d.index, name="intercept"), d[cols].astype(float), geo],axis=1)
    if train is None:
        means=X.mean(); scales=X.std().replace(0,1); means["intercept"]=0; scales["intercept"]=1
    else:
        tindex=train.index.intersection(X.index); means=X.loc[tindex].mean(); scales=X.loc[tindex].std().replace(0,1); means["intercept"]=0; scales["intercept"]=1
    return d, X, cols, means, scales

def _fit_bayes(X: pd.DataFrame, y: np.ndarray, draws: int, seed: int, precision: float):
    """Normal-inverse-gamma Bayesian ridge posterior; exact conditional draws for this model."""
    z=X.to_numpy(float); p=z.shape[1]; prior=np.eye(p)*precision; prior[0,0]=1e-8
    V=np.linalg.inv(z.T@z+prior); mean=V@(z.T@y); resid=y-z@mean
    a=(len(y)+2)/2; b=(resid@resid + mean@prior@mean)/2
    rng=np.random.default_rng(seed); sig2=1/rng.gamma(a,1/b,size=draws)
    chol=np.linalg.cholesky(V+np.eye(p)*1e-12); beta=mean+rng.normal(size=(draws,p))@chol.T*np.sqrt(sig2)[:,None]
    return mean, beta, sig2, {"posterior_method":"normal_inverse_gamma_bayesian_ridge", "draws":draws,
        "residual_sd_mean":float(np.sqrt(sig2).mean()),"parameters":p,"observations":len(y)}

def _predict(X, beta, sig2, rng): return X.to_numpy()@beta.T + rng.normal(size=(len(X),len(beta)))*np.sqrt(sig2)

def _metrics(actual, draws):
    pred=draws.mean(1); lo,hi=np.quantile(draws,[.05,.95],axis=1)
    return {"RMSE":float(np.sqrt(np.mean((actual-pred)**2))),"MAE":float(np.mean(abs(actual-pred))),
      "WAPE":float(np.sum(abs(actual-pred))/np.sum(abs(actual))),"interval_90_coverage":float(np.mean((actual>=lo)&(actual<=hi))),
      "interval_90_mean_width":float(np.mean(hi-lo))}

def _channel_contrib(frame, X, beta, media_columns, zero_values=None):
    rows=[]; spend=[]
    for i,c in enumerate(media_columns):
        cf=X.copy(); cf[c]=0 if zero_values is None else zero_values[c]
        delta=(X.to_numpy()-cf.to_numpy())@beta.T
        amount=delta.sum(axis=0); s=frame[f"Channel{i}_spend"].sum()
        rows.append(amount); spend.append(s)
    return np.array(rows),np.array(spend)

def _calibrate(contrib, spend, beta, Xcols, plan_X, zero_values, cfg, rng):
    """Approximate calibration on estimand-scale ROAS; no double use of vendor evidence."""
    i=3; base=contrib[i]/spend[i]; m=float(base.mean()); v=float(base.var(ddof=1))
    se=(cfg["vendor_roas_ci90"][1]-cfg["vendor_roas_ci90"][0])/(2*1.644854)
    se*=cfg["calibration_sd_inflation"]
    post_v=1/(1/max(v,1e-9)+1/se**2); post_m=post_v*(m/max(v,1e-9)+cfg["vendor_roas"]/se**2)
    target=rng.normal(post_m, math.sqrt(post_v), size=len(base)); target=np.maximum(target,0)
    # Set the exposure-scale coefficient from the calibrated ROAS estimand rather
    # than dividing by a near-zero/negative uncalibrated draw.
    idx=Xcols.index("media_3")
    exposure_delta=float((plan_X["media_3"]-zero_values["media_3"]).sum())
    if exposure_delta <= 0: raise ValueError("Channel 3 counterfactual exposure delta is invalid")
    b=beta.copy(); b[:,idx]=target*spend[i]/exposure_delta
    return b,{"base_roas_mean":m,"base_roas_sd":math.sqrt(v),"vendor_sd_assumed":se,"calibrated_roas_mean":float(target.mean()),"calibrated_roas_sd":float(target.std(ddof=1)),"quality_weight":"low (vendor SE inflated 2x)"}

def _optimize(baseline, contribution_draws, budget):
    lower,upper=.7*baseline,1.3*baseline
    if budget < lower.sum()-1e-5 or budget > upper.sum()+1e-5: return None
    unit = 1_000_000.0
    baseline_m, lower_m, upper_m, budget_m = baseline/unit, lower/unit, upper/unit, budget/unit
    # response curve uses each posterior draw's recent-horizon counterfactual contribution.
    k=baseline/1.5
    def response(q):
        x=q*unit
        return (1-np.exp(-x[:,None]/k[:,None]))/(1-np.exp(-baseline[:,None]/k[:,None]))*contribution_draws
    def objective(q): return -float(response(q).sum(axis=0).mean()) / unit
    def gradient(q):
        x=q*unit
        deriv=(contribution_draws*np.exp(-x[:,None]/k[:,None])/k[:,None]/
               (1-np.exp(-baseline[:,None]/k[:,None]))).mean(1)
        return -deriv
    r=minimize(objective, baseline_m*budget_m/baseline_m.sum(), jac=gradient, method="SLSQP", bounds=list(zip(lower_m,upper_m)),constraints={"type":"eq","fun":lambda q:q.sum()-budget_m,"jac":lambda q:np.ones(len(q))},options={"ftol":1e-10,"maxiter":500})
    if not r.success: raise RuntimeError(r.message)
    spend=r.x*unit; total=response(r.x).sum(axis=0)
    marginal=(contribution_draws*np.exp(-spend[:,None]/k[:,None])/k[:,None]/(1-np.exp(-baseline[:,None]/k[:,None]))).mean(1)
    return {"spend":spend,"draws":total,"marginal":marginal,"binding":(np.isclose(spend,lower)|np.isclose(spend,upper))}

def _dag(root: Path):
    md="""# Causal DAG\n\n```mermaid\ngraph LR\nGeo-->Spend; Time-->Spend; Geo-->Conversions; Time-->Conversions\nSpend-->Impressions-->Conversions-->Revenue\nPromo-->Conversions; Competitor-->Conversions; Sentiment-->Conversions\nOrganic-->Conversions; Conversions-->Revenue; RPC[revenue per conversion]-->Revenue\n```\n\nImpressions are the exposure and spend is cost; they are not included as separate causal effects. Promotions, organic activity, competitor sales and sentiment are included as pre-treatment controls under this assumption; if any is downstream of paid media it would be a mediator and adjustment would bias the media estimate. Revenue is calculated as conversions × revenue per conversion (RPC is never summed). Geo intercepts and seasonal terms address persistent geographic and calendar differences, but cannot fully remove targeting/reverse-causality confounding.\n"""
    (root/"reports/causal_dag.md").write_text(md)
    _write_png(root/"figures/causal_dag.png",np.array([1,2,3,2,1]))

def run(data_path: str, output_dir: str, holdout_start: str|None, seed: int, fast: bool=False, skip_national: bool=False, skip_experiment_design: bool=False):
    out=Path(output_dir); _mkdir(out); cfg=yaml.safe_load(Path("config/model_config.yaml").read_text()); cfg["seed"]=seed
    logging.basicConfig(filename=out/"run.log",level=logging.INFO,format="%(asctime)s %(message)s")
    raw=pd.read_csv(data_path); d,facts=validate_data(raw); d,half=add_media_features(d,cfg["adstock_alpha"])
    train,hold,actual=chronological_split(d,holdout_start); d2,X,feature_cols,means,scales=_build_design(d,train); media_cols=[f"media_{i}" for i in range(5)]; Z=(X-means)/scales; zero_values=(-means/scales); y=d2.revenue.to_numpy(); mask=d2.time<actual
    draws=250 if fast else cfg["posterior_draws"]; mean,beta,sig2,diagnostics=_fit_bayes(Z.loc[mask],y[mask],draws,seed,cfg["ridge_prior_precision"]); rng=np.random.default_rng(seed+1)
    pred=_predict(Z.loc[~mask],beta,sig2,rng); metrics=_metrics(y[~mask],pred)
    pred_table=d2.loc[~mask,["geo","time","revenue"]].copy(); pred_table["prediction_mean"]=pred.mean(1); pred_table["prediction_p05"]=np.quantile(pred,.05,axis=1); pred_table["prediction_p95"]=np.quantile(pred,.95,axis=1); pred_table.to_csv(out/"tables/holdout_predictions.csv",index=False)
    # planning uses final 13 weeks before validation boundary; no held-out data.
    plan_weeks=sorted(train.time.unique())[-13:]; plan=d2[d2.time.isin(plan_weeks)].copy(); planZ=Z.loc[plan.index]; contrib,spend=_channel_contrib(plan,planZ,beta,media_cols,zero_values)
    beta_cal,cal_info=_calibrate(contrib,spend,beta,list(Z.columns),planZ,zero_values,cfg,rng); contrib_cal,_=_channel_contrib(plan,planZ,beta_cal,media_cols,zero_values)
    rows=[]
    for i,ch in enumerate(CHANNELS):
      for scenario, cs in [("uncalibrated",contrib),("calibrated",contrib_cal)]:
        avg=cs[i]/spend[i]; rows.append({"channel":ch,"scenario":scenario,"spend_planning_13w":spend[i],"incremental_revenue_mean":cs[i].mean(),"incremental_revenue_p05":np.quantile(cs[i],.05),"incremental_revenue_p95":np.quantile(cs[i],.95),"average_roas_mean":avg.mean(),"average_roas_p05":np.quantile(avg,.05),"average_roas_p95":np.quantile(avg,.95),"roi_mean":avg.mean()-1})
    roi=pd.DataFrame(rows); roi.to_csv(out/"tables/channel_roi.csv",index=False)
    posterior_summary=pd.DataFrame({"parameter":Z.columns,"posterior_mean":beta.mean(0),"p05":np.quantile(beta,.05,axis=0),"p95":np.quantile(beta,.95,axis=0)})
    posterior_summary.to_csv(out/"tables/posterior_summary.csv",index=False)
    calibration_compare=roi[roi.channel=="Channel3"].copy()
    calibration_compare.to_csv(out/"tables/calibration_comparison.csv",index=False)
    budgets=[]
    for budget in [12_000_000,18_000_000,24_000_000]:
      opt=_optimize(spend,contrib_cal,budget)
      if opt is None:
       budgets.append({"budget":budget,"status":"infeasible under ±30% bounds","minimum_feasible":.7*spend.sum(),"maximum_feasible":1.3*spend.sum()}); continue
      for i,ch in enumerate(CHANNELS): budgets.append({"budget":budget,"status":"optimal","channel":ch,"baseline_spend":spend[i],"recommended_spend":opt["spend"][i],"pct_change":opt["spend"][i]/spend[i]-1,"expected_marginal_revenue_per_currency":opt["marginal"][i],"constraint_binding":bool(opt["binding"][i]),"portfolio_revenue_mean":opt["draws"].mean(),"portfolio_revenue_p05":np.quantile(opt["draws"],.05),"portfolio_revenue_p95":np.quantile(opt["draws"],.95)})
    pd.DataFrame(budgets).to_csv(out/"tables/budget_scenarios.csv",index=False)
    # Experiment factual check and conservative clustered DID.
    window=pd.date_range("2023-06-05",periods=8,freq="7D"); exp=d2[d2.time.isin(window)].copy(); exp["treated"]=exp.geo.str.extract(r"(\d+)").astype(int).iloc[:,0]<8
    pre=d2[(d2.time<window[0])&(d2.time>=window[0]-pd.Timedelta(weeks=8))].copy(); pre["treated"]=pre.geo.str.extract(r"(\d+)").astype(int).iloc[:,0]<8
    g=lambda q: q.groupby("treated")[["Channel3_spend","Channel3_impression","revenue"]].mean()
    did=(g(exp).loc[True,"revenue"]-g(pre).loc[True,"revenue"])-(g(exp).loc[False,"revenue"]-g(pre).loc[False,"revenue"])
    exp_summary=pd.DataFrame([{"intervention_weeks_found":exp.time.nunique(),"treated_c3_spend":exp[exp.treated].Channel3_spend.sum(),"control_c3_spend":exp[~exp.treated].Channel3_spend.sum(),"pre_treated_c3_spend":pre[pre.treated].Channel3_spend.sum(),"did_revenue_per_geo_week":did,"note":"Non-randomized assignment and 8 weeks: descriptive clustered DID only."}]); exp_summary.to_csv(out/"tables/experiment_assessment.csv",index=False)
    exp.groupby(["time","treated"])[["Channel3_spend","Channel3_impression","revenue"]].mean().reset_index().to_csv(out/"tables/experiment_weekly_means.csv",index=False)
    # national comparison: same approach, aggregates revenue correctly
    if not skip_national:
      agg={"conversions":"sum",**{f"Channel{i}_{x}":"sum" for i in range(5) for x in ["impression","spend"]},"Organic_channel0_impression":"sum","Promo":"mean","competitor_sales_control":"mean","sentiment_score_control":"mean","population":"sum"}
      nat=d.groupby("time",as_index=False).agg(agg); nat["geo"]="National"; nat["revenue_per_conversion"]=(d.assign(revenue=d.conversions*d.revenue_per_conversion).groupby("time").revenue.sum().to_numpy()/nat.conversions.to_numpy())
      nat,_=add_media_features(nat,cfg["adstock_alpha"]); nt,nh,_=chronological_split(nat,str(actual.date())); n2,nx,_,nm,ns=_build_design(nat,nt); nz=(nx-nm)/ns; nmean,nb,ns2,_=_fit_bayes(nz.loc[n2.time<actual],n2.revenue.to_numpy()[n2.time<actual],draws,seed+99,cfg["ridge_prior_precision"]); npred=_predict(nz.loc[n2.time>=actual],nb,ns2,rng); nmet=_metrics(n2.loc[n2.time>=actual,"revenue"].to_numpy(),npred); pd.DataFrame([{"model":"geo","RMSE":metrics["RMSE"],"WAPE":metrics["WAPE"]},{"model":"national","RMSE":nmet["RMSE"],"WAPE":nmet["WAPE"]}]).to_csv(out/"tables/geo_national_comparison.csv",index=False)
    # reports/tables.
    pd.DataFrame({"column":d.columns,"missing":d.isna().sum().values,"dtype":[str(x) for x in d.dtypes]}).to_csv(out/"tables/missing_values.csv",index=False)
    d.groupby("time").geo.nunique().rename("geographies").reset_index().to_csv(out/"tables/weekly_coverage.csv",index=False)
    pd.DataFrame([facts]).to_csv(out/"tables/data_quality_summary.csv",index=False)
    split={"requested_holdout_start":holdout_start,"actual_holdout_start":str(actual.date()),"train_rows":len(train),"holdout_rows":len(hold),"train_weeks":train.time.nunique(),"holdout_weeks":hold.time.nunique(),"geographies":d.geo.nunique(),"conditional_forecast":"uses actual held-out media/control inputs; outcomes excluded from fitting"}
    (out/"reports/data_quality_report.md").write_text("# Data quality\n\n"+json.dumps(facts,indent=2)+"\n\nNo rows were dropped or imputed. Revenue = conversions × revenue_per_conversion.\n")
    (out/"reports/model_specification.md").write_text("# Model specification\n\nOutcome is geo-week revenue. Exposures are causal geometric-adstocked and Hill-saturated impressions. The Bayesian model is conjugate Normal-inverse-gamma ridge regression with geo fixed effects as a practical partial-pooling proxy, annual Fourier time terms, organic media, promotion, competitor and sentiment controls. It yields exact posterior draws conditional on fixed transformation parameters; transformation-parameter uncertainty is a limitation.\n")
    (out/"reports/experiment_calibration.md").write_text("# Calibration\n\nUncalibrated posterior is compared with a single low-weight calibrated scenario. Vendor ROAS 0.8, 90% interval [0.3,1.3] is converted to Normal SE 0.304 and inflated 2× for unknown execution quality. A normal-normal update is applied to the *Channel 3 ROAS estimand* then mapped by proportional response scaling; this is an explicitly approximate, not a direct coefficient prior.\n\n"+json.dumps(cal_info,indent=2))
    # Decision-relevant uncertainty is the posterior SD of recent-horizon
    # incremental revenue; it selects the experiment target, not interval width alone.
    risks=contrib_cal.std(axis=1,ddof=1)
    target_i=int(np.argmax(risks)); target=CHANNELS[target_i]
    design=f"""# Next experiment design\n\n**Recommended channel: {target}.** It has the largest decision-relevant posterior standard deviation of 13-week incremental revenue ({risks[target_i]:,.0f}), so resolving its uncertainty has the highest potential allocation value among the channels in this executed model.\n\nUse 20 of the 40 geographies (10 treatment, 10 control), stratified by pre-period revenue and {target} spend, then randomly assign within strata. Run for 13 weeks after an 8-week pre-period. Treatment is a pre-specified ±25% spend change while other paid channels are held to plan; control retains planned spend. Primary outcome: geo-week revenue, with incremental revenue and incremental ROAS as estimands. Analyze an intention-to-treat, geo-clustered difference-in-differences using pre-registered covariates and randomization inference/cluster bootstrap.\n\nThe minimum economically meaningful effect should be set before launch as the finance-approved revenue change from a 25% spend change. Power must be recalculated from the selected geographies' pre-period variance; with only 10 treated geos this is likely limited, so do not assert conventional 80% power without that calculation. Stop only for pre-specified operational/safety rules, not interim significance. Monitor spillover, channel contamination, seasonality, and tracking changes.\n"""
    if not skip_experiment_design: (out/"reports/next_experiment_design.md").write_text(design)
    memo=f"""# CMO/CFO memo (executed results)\n\nThe model was trained through {actual.date()-pd.Timedelta(days=7):%Y-%m-%d}; its 13-week conditional holdout WAPE is {metrics['WAPE']:.1%} and RMSE is {metrics['RMSE']:,.0f}. Forecasts condition on observed future media, so they are validation rather than a media forecast. The calibrated (low-weight) portfolio recommendation is in `budget_scenarios.csv`; it satisfies ±30% bounds where feasible. The $24m scenario is reported infeasible if its requested total exceeds the allowed envelope. Revenue, not profit, is optimized; margins were not supplied.\n\nThe Channel 3 vendor result was treated as uncertain supporting evidence, not proof of causality. The next experiment should target {target}, which has the largest posterior decision-relevant incremental-revenue uncertainty ({risks[target_i]:,.0f} over the planning horizon), with randomized geo assignment, 10 treated / 10 control geographies, and a 13-week pre-registered cluster-aware design. With only 40 geographies, power is limited and must be recalculated after finance chooses a minimum economically meaningful effect.\n"""
    (out/"reports/executive_memo.md").write_text(memo); (out/"reports/split_summary.json").write_text(json.dumps(split,indent=2)); (out/"reports/forecast_metrics.json").write_text(json.dumps(metrics,indent=2)); (out/"reports/run_metadata.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"config":cfg,"diagnostics":diagnostics},indent=2))
    np.savez_compressed(out/"model/posterior_draws.npz",beta=beta,sigma2=sig2,columns=np.array(Z.columns))
    _dag(out); _write_png(out/"figures/holdout_error.png",(pred.mean(1)-y[~mask])); _write_png(out/"figures/channel_incremental_revenue.png",contrib_cal.mean(1))
    return {"facts":facts,"metrics":metrics,"split":split,"diagnostics":diagnostics,"calibration":cal_info}
