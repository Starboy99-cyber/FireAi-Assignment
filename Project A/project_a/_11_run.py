from __future__ import annotations
import argparse, json, logging, os, platform, time, textwrap
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config, ASSUMPTIONS
from .data import discover_paths, load_scope, input_metadata
from .validation import validate_scope, price_coverage
from .censoring import diagnostics, scenarios
from .forecasting import seasonal_quantile_forecast, moving_average_forecast, to_frame, validate_forecasts
from .evaluation import fold_rows, summarize_policy
from .simulate import simulate
from .reconciliation import reconcile_scenarios

def _write_svg(path: Path, title: str, lines: list[str]):
    escaped=[str(x).replace('&','and').replace('<','') for x in lines]
    body=''.join(f'<text x="30" y="{70+i*26}" font-family="Arial" font-size="16">{x}</text>' for i,x in enumerate(escaped))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{max(180,110+30*len(lines))}"><rect width="100%" height="100%" fill="white"/><text x="30" y="38" font-family="Arial" font-size="24" font-weight="bold">{title}</text>{body}</svg>',encoding='utf-8')

def _pdf_escape(s): return s.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
def write_memo_pdf(path: Path, summary: dict):
    """Minimal standards-compliant, two-page text PDF with no dependency on a GUI renderer."""
    pol=summary.get('policy',{}); stress=summary.get('stress',{}); lines=[
      'Fire.Ai Project A - COO Memo', '', 'Recommendation', 'Select the base-stock policy for the current rollout. The forecast-driven candidate needs more tuning because it did not beat the baseline.',
      f"Final observed-sales proxy: proposed fill rate {pol.get('proposed_fill_rate',float('nan')):.1%}; base-stock {pol.get('baseline_fill_rate',float('nan')):.1%}.",
      f"Final simulated cost: proposed ${pol.get('proposed_cost',float('nan')):,.0f}; base-stock ${pol.get('baseline_cost',float('nan')):,.0f}.",
      f"95% target achieved? {'Yes' if pol.get('proposed_fill_rate',0)>=.95 else 'No - disclose shortfall and refine before rollout.'}", '',
      'How the policy works', 'It uses empirical weekday forecast quantiles, lead-time protection, daily store transfer requests, Monday-only external POs,', 'and hard DC/store capacity limits. DC quantiles are computed from coherent demand scenarios, not summed marginal quantiles.',
      'Observed sales are a proxy for demand: the source has no inventory or verified stockout labels. Recovery cases are sensitivities only.', '',
      'Stress test', f"FOODS_3 doubled-lead-time disruption: fill rate {stress.get('disruption_fill_rate',float('nan')):.1%}; warning mitigation {stress.get('mitigation_fill_rate',float('nan')):.1%}.",
      'Mitigation pre-positions only during the permitted two-week warning and respects the same capacities.', '',
      'Three risks and data needed', '1. No inventory/stockout observations means censored demand is not identifiable. Collect on-hand, availability and lost-sales signals.', '2. Missing historical prices constrain cost accuracy. Collect listing-effective dates and complete price history.', '3. Store demand dependence may differ from independent scenario assumptions. Add promotion, substitution and DC order data.',
      'All figures are simulated policy outcomes, not observed business savings.' ]
    wrapped=[]
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=105, break_long_words=False) or [''])
    chunks=[wrapped[:23], [lines[0]+' (continued)','']+wrapped[23:]]; objs=[]
    for page in chunks:
        content='BT /F1 9 Tf 45 750 Td 13 TL ' + ' '.join(f'({_pdf_escape(l)}) Tj T*' for l in page) + ' ET'
        objs.append(content.encode())
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>', b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>', b'', b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>', b'', b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    for index,data in ((3,objs[0]),(5,objs[1])): objects[index]=b'<< /Length '+str(len(data)).encode()+b' >>\nstream\n'+data+b'\nendstream'
    out=bytearray(b'%PDF-1.4\n'); offsets=[0]
    for i,obj in enumerate(objects,1): offsets.append(len(out)); out+=f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
    start=len(out); out+=f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()+b''.join(f'{x:010d} 00000 n \n'.encode() for x in offsets[1:])+f'trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode(); path.write_bytes(out)

def write_docs(root: Path, summary: dict):
    (root/'DATA_ISSUES.md').write_text('''# Data issues\n\n- Sales are observed sales, not latent demand. No inventory, availability, or stockout labels establish that zero sales mean stockouts.\n- Missing prices are retained with an indicator; a transparent median fallback is used only for simulated monetary costs.\n- Sparse/inactive series may represent low demand, discontinuation, availability gaps, or assortment differences.\n- Cross-store activity is suggestive only; it is not stockout evidence. Calendar coverage ends at d_1941 and demand may drift.\n''')
    (root/'DECISIONS.md').write_text('''# Decisions\n\nInputs are resolved as exact files or CSV/Parquet extensions and loaded read-only. Scope is CA_1--CA_4 and FOODS_1--FOODS_3. Lags/rolls are shifted; final d_1886--d_1941 is untouched until final evaluation.\n\nThe selected candidate is a comparable-weekday empirical quantile baseline when rolling WAPE supports it; moving-average is retained as a comparator. Zeros are not labeled stockouts. Conservative/higher recovery are flagged-series sensitivities. DC quantiles are derived from sampled store scenarios under an independent conditional-demand assumption.\n\nThe simulator uses arrivals, sales/lost sales, holding, transfer decisions, then Monday POs. Continuous lognormal lead times are rounded up to positive days. Scarce outbound capacity is allocated by forecast-demand urgency. Store caps and Monday ordering are hard constraints. Initial stores hold 7 days average demand; DC starts at a configurable 14 days. Final policy is frozen after rolling validation.\n''')
    (root/'AI_USAGE.md').write_text('''# AI usage\n\nAI assistance was used to draft project code, documentation, and test cases. The implementation executed its data pipeline and automated checks in this environment; generated metrics in the outputs are programmatically produced. No claim is made of human manual review beyond those checks.\n''')
    (root/'README.md').write_text('''# Fire.Ai Project A\n\nReproducible inventory-policy analysis for censored retail sales. Install `pip install -r requirements.txt`, then run `python -m project_a.run --help` or `run_project_a.bat`. Inputs default to `C:\\Users\\niraj\\Downloads\\Data`; the final holdout is d_1886--d_1941.\n\n`outputs/` contains validation, forecasts, policy and stress-test results; `deliverables/` contains the memo and limitations. Use `--max-series` only for disclosed quick development runs. Run checks with `python -m unittest discover -s tests -v`.\n''')

def run(cfg: Config):
    started=time.time(); root=Path.cwd(); out=(root/cfg.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); (root/'logs').mkdir(exist_ok=True); (root/'deliverables').mkdir(exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    logging.basicConfig(filename=root/'logs'/'run.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s'); log=logging.getLogger(__name__)
    paths=discover_paths(cfg.sales_path,cfg.prices_path,cfg.calendar_path); print('Resolved input paths:', {k:str(v) for k,v in paths.items()}); log.info('Resolved input paths %s',paths)
    meta,days,values,calendar,prices,raw_shape=load_scope(paths,cfg.stores,cfg.departments,cfg.max_series)
    price,price_stats=price_coverage(meta,values,calendar,prices,cfg.cutoff+56); validation=validate_scope(meta,raw_shape,values,calendar,price_stats); validation['input_files']=input_metadata(paths,{'sales':pd.DataFrame(columns=meta.columns),'prices':prices.head(0),'calendar':calendar.head(0)})
    (out/'validation_report.json').write_text(json.dumps(validation,indent=2,default=str)); diag=diagnostics(meta,values,cfg.cutoff); diag.to_csv(out/'censoring_diagnostics.csv',index=False); scenarios(diag).to_csv(out/'demand_recovery_scenarios.csv',index=False)
    metrics=[]; selected='seasonal_empirical';
    for fold,(origin,lo,hi) in enumerate(cfg.rolling_folds,1):
        actual=values[:,lo-1:hi];
        for name,fun in [('seasonal_empirical',seasonal_quantile_forecast),('moving_average',moving_average_forecast)]: metrics.extend(fold_rows(meta,actual,fun(values,origin,hi-lo+1),fold,name))
    mf=pd.DataFrame(metrics); mf.to_csv(out/'forecast_metrics.csv',index=False); mf[mf.dept_id.notna() if 'dept_id' in mf else pd.Series(False,index=mf.index)].to_csv(out/'forecast_metrics_by_department.csv',index=False)
    point=mf[(mf["quantile"]=='point') & mf["day_ahead"].isna()]; means=point.groupby('model')["wape"].mean(); selected=str(means.idxmin())
    final_fun=seasonal_quantile_forecast if selected=='seasonal_empirical' else moving_average_forecast; final=final_fun(values,cfg.cutoff,56); ff=to_frame(meta,final,cfg.cutoff,model=selected); validate_forecasts(ff); ff.to_parquet(out/'forecasts.parquet',index=False)
    dc_scenarios,dc_q=reconcile_scenarios(ff,seed=cfg.seed,draws=50); dc_q.to_csv(out/'dc_forecast_quantiles.csv',index=False); dc_scenarios.head(200000).to_parquet(out/'dc_demand_scenarios_sample.parquet',index=False)
    actual=values[:,cfg.cutoff:cfg.cutoff+56]; mean=values[:,:cfg.cutoff].mean(axis=1); fq=final[.90]; final_prices=price[:,cfg.cutoff:cfg.cutoff+56]; records=[]; traces=[]
    for proposed,name in ((False,'base_stock'),(True,'forecast_driven')):
      for seed in range(cfg.seed,cfg.seed+cfg.replications):
        r,t=simulate(meta,actual,fq,final_prices,mean,seed,proposed); r.update({'period':'final_holdout','seed':seed,'policy_name':name});records.append(r);t['policy_name']=name;t['seed']=seed;traces.append(t)
    # Rolling policy evaluation is frozen default policy; not tuned on holdout.
    for fold,(origin,lo,hi) in enumerate(cfg.rolling_folds,1):
      f=final_fun(values,origin,hi-lo+1); a=values[:,lo-1:hi]; p=price[:,lo-1:hi]; m=values[:,:origin].mean(axis=1)
      for proposed,name in ((False,'base_stock'),(True,'forecast_driven')):
        r,_=simulate(meta,a,f[.90],p,m,cfg.seed+fold,proposed);r.update({'period':f'fold_{fold}','seed':cfg.seed+fold,'policy_name':name});records.append(r)
    base, _=simulate(meta,actual,fq,final_prices,mean,cfg.seed,False); dis,_=simulate(meta,actual,fq,final_prices,mean,cfg.seed,True,'disruption',False); mitigation,_=simulate(meta,actual,fq,final_prices,mean,cfg.seed,True,'disruption',True)
    normal_record=next(r for r in records if r["period"]=='final_holdout' and r["seed"]==cfg.seed and r["policy_name"]=='forecast_driven')
    st=pd.DataFrame([{'scenario':'normal','policy':'forecast_driven',**normal_record},{'scenario':'disruption_unwarned','policy':'forecast_driven',**dis},{'scenario':'disruption_two_week_warning','policy':'forecast_driven',**mitigation}]); st.to_csv(out/'stress_test_results.csv',index=False)
    policy=summarize_policy(records); policy.to_csv(out/'policy_comparison.csv',index=False); trace=pd.concat(traces,ignore_index=True); trace.to_csv(out/'inventory_trace.csv',index=False); pd.DataFrame(records).to_csv(out/'simulation_runs.csv',index=False)
    normal=st.iloc[0]; summary={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'config':{'cutoff':cfg.cutoff,'seed':cfg.seed,'replications':cfg.replications,'selected_model':selected},'validation':validation,'forecast_model_wape':means.to_dict(),'policy':{'baseline_cost':float(base['total_cost']),'baseline_fill_rate':float(base['fill_rate']),'proposed_cost':float(normal.total_cost),'proposed_fill_rate':float(normal.fill_rate)},'stress':{'disruption_fill_rate':float(dis['fill_rate']),'mitigation_fill_rate':float(mitigation['fill_rate'])},'runtime_seconds':time.time()-started,'warnings':['Observed sales are not verified demand; no zero-sales day is classified as a stockout.','Price fallback affects monetary simulation where historical price is missing.']}
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2,default=str)); _write_svg(out/'figures'/'forecast_calibration.svg','Forecast calibration',[f'{x.model}: rolling WAPE {x.wape:.4f}' for x in point.groupby('model').mean(numeric_only=True).reset_index().itertuples()]); _write_svg(out/'figures'/'policy_comparison.svg','Policy results',[f"{x.policy_name} {x.period}: cost ${x.total_cost:,.0f}, fill {x.fill_rate:.1%}" for x in policy.itertuples()]); _write_svg(out/'figures'/'censoring_diagnostics.svg','Zero-sales diagnostics',[f"series {len(diag)}; median zero rate {diag.zero_sales_rate.median():.1%}", 'Flags are sensitivities, not stockout labels.']); _write_svg(out/'figures'/'stress_test.svg','FOODS_3 lead-time stress test',[f"{x.scenario}: cost ${x.total_cost:,.0f}; fill {x.fill_rate:.1%}" for x in st.itertuples()])
    write_docs(root,summary); write_memo_pdf(root/'deliverables'/'memo_A.pdf',summary)
    print(json.dumps({'selected_model':selected,'runtime_seconds':summary['runtime_seconds'],'outputs':str(out)},indent=2)); return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--sales-path',default=str(Config.sales_path));p.add_argument('--prices-path',default=str(Config.prices_path));p.add_argument('--calendar-path',default=str(Config.calendar_path));p.add_argument('--output-dir',default='outputs');p.add_argument('--cutoff',default='d_1885');p.add_argument('--seed',type=int,default=42);p.add_argument('--max-series',type=int);p.add_argument('--replications',type=int,default=3);a=p.parse_args(); cutoff=int(str(a.cutoff).replace('d_','')); run(Config(Path(a.sales_path),Path(a.prices_path),Path(a.calendar_path),Path(a.output_dir),cutoff,a.seed,a.max_series,a.replications))
if __name__=='__main__': main()
