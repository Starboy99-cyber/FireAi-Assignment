# Model specification

Outcome is geo-week revenue. Exposures are causal geometric-adstocked and Hill-saturated impressions. The Bayesian model is conjugate Normal-inverse-gamma ridge regression with geo fixed effects as a practical partial-pooling proxy, annual Fourier time terms, organic media, promotion, competitor and sentiment controls. It yields exact posterior draws conditional on fixed transformation parameters; transformation-parameter uncertainty is a limitation.
