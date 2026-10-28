# Calibration

Uncalibrated posterior is compared with a single low-weight calibrated scenario. Vendor ROAS 0.8, 90% interval [0.3,1.3] is converted to Normal SE 0.304 and inflated 2× for unknown execution quality. A normal-normal update is applied to the *Channel 3 ROAS estimand* then mapped by proportional response scaling; this is an explicitly approximate, not a direct coefficient prior.

{
  "base_roas_mean": 0.826559242324415,
  "base_roas_sd": 0.4852369541929662,
  "vendor_sd_assumed": 0.607956694028771,
  "calibrated_roas_mean": 0.8040460713679994,
  "calibrated_roas_sd": 0.37829398154335775,
  "quality_weight": "low (vendor SE inflated 2x)"
}