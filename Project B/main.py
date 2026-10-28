from __future__ import annotations
import argparse, json, sys
from src.pipeline import run
def main():
 p=argparse.ArgumentParser(description="Geo-level Bayesian MMM with experiment calibration")
 p.add_argument("--data-path",default="data/raw/geo_all_channels.csv"); p.add_argument("--output-dir",default="outputs")
 p.add_argument("--holdout-start",default=None,help="YYYY-MM-DD; defaults to final 13 observed weeks")
 p.add_argument("--seed",type=int,default=20261028); p.add_argument("--fast",action="store_true",help="Use 250 instead of 800 posterior draws; development only")
 p.add_argument("--skip-national-model",action="store_true"); p.add_argument("--skip-experiment-design",action="store_true")
 p.add_argument("--sampled-mcmc",action="store_true",help="Run the PyMC hierarchical sampled model (requires the project .venv)")
 p.add_argument("--mcmc-draws",type=int,default=300); p.add_argument("--mcmc-tune",type=int,default=300)
 a=p.parse_args()
 if a.sampled_mcmc:
  from src.gibbs_model import run_gibbs
  result=run_gibbs(a.data_path,a.output_dir,a.holdout_start,a.seed,a.mcmc_draws,a.mcmc_tune)
 else: result=run(a.data_path,a.output_dir,a.holdout_start,a.seed,a.fast,a.skip_national_model,a.skip_experiment_design)
 print(json.dumps(result,indent=2))
if __name__=="__main__":
 try: main()
 except Exception as exc: print(f"PIPELINE FAILED: {exc}",file=sys.stderr); raise
