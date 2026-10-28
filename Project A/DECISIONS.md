# Decisions

Inputs are resolved as exact files or CSV/Parquet extensions and loaded read-only. Scope is CA_1--CA_4 and FOODS_1--FOODS_3. Lags/rolls are shifted; final d_1886--d_1941 is untouched until final evaluation.

The selected candidate is a comparable-weekday empirical quantile baseline when rolling WAPE supports it; moving-average is retained as a comparator. Zeros are not labeled stockouts. Conservative/higher recovery are flagged-series sensitivities. DC quantiles are derived from sampled store scenarios under an independent conditional-demand assumption.

The simulator uses arrivals, sales/lost sales, holding, transfer decisions, then Monday POs. Continuous lognormal lead times are rounded up to positive days. Scarce outbound capacity is allocated by forecast-demand urgency. Store caps and Monday ordering are hard constraints. Initial stores hold 7 days average demand; DC starts at a configurable 14 days. Final policy is frozen after rolling validation.
