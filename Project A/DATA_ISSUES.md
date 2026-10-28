# Data issues

- Sales are observed sales, not latent demand. No inventory, availability, or stockout labels establish that zero sales mean stockouts.
- Missing prices are retained with an indicator; a transparent median fallback is used only for simulated monetary costs.
- Sparse/inactive series may represent low demand, discontinuation, availability gaps, or assortment differences.
- Cross-store activity is suggestive only; it is not stockout evidence. Calendar coverage ends at d_1941 and demand may drift.
