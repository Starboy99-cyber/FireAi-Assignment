# Fire.Ai Project A

Reproducible inventory-policy analysis for censored retail sales. Install `pip install -r requirements.txt`, then run `python -m project_a.run --help` or `run_project_a.bat`. Inputs default to `C:\Users\niraj\Downloads\Data`; the final holdout is d_1886--d_1941.

`outputs/` contains validation, forecasts, policy and stress-test results; `deliverables/` contains the memo and limitations. Use `--max-series` only for disclosed quick development runs. Run checks with `python -m unittest discover -s tests -v`.
