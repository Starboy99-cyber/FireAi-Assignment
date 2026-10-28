# Project B — Geo-level MMM

Executed project using the official Google Meridian `geo_all_channels.csv` data. The pipeline validates the 6,240 geo-week input, constructs revenue as `conversions * revenue_per_conversion`, fits a reproducible Bayesian conjugate ridge MMM with causal adstock/Hill media features, validates a chronological 13-week holdout, evaluates the stated Channel 3 intervention, compares a low-weight experiment-calibrated scenario, and produces constrained budget scenarios.

Run from the repository root:

```powershell
python main.py
```

For the sampled hierarchical MCMC run used for the upgraded diagnostics:

```powershell
.\.venv\Scripts\python.exe main.py --sampled-mcmc --output-dir outputs_mcmc --mcmc-draws 800 --mcmc-tune 800
```

Optional: `python main.py --holdout-start 2023-10-23 --output-dir outputs_alt --fast`.

`--fast` is only a development run (250 posterior draws); the normal run uses 800. Forecast validation is conditional on observed holdout media/control inputs. The sampled run uses a two-chain Gibbs sampler with hierarchical geo intercepts and reports real MCMC diagnostics. Its adstock and Hill saturation parameters remain fixed because joint PyMC estimation needs a C++ compiler unavailable on this machine. See `outputs/reports/` and `outputs_mcmc/reports/` for assumptions and limitations.
