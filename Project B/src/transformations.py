"""Media transformations used consistently by training and scenarios."""
from __future__ import annotations
import numpy as np
import pandas as pd

def geometric_adstock(values: np.ndarray, alpha: float) -> np.ndarray:
    """Causal geometric adstock; value at t only uses observations through t."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1).")
    result = np.zeros(len(values), dtype=float)
    for i, value in enumerate(values):
        result[i] = value + (alpha * result[i - 1] if i else 0.0)
    return result

def hill_saturation(values: np.ndarray, half_saturation: float, slope: float = 1.0) -> np.ndarray:
    """Stable Hill curve bounded in [0, 1]."""
    if half_saturation <= 0 or slope <= 0:
        raise ValueError("Hill parameters must be positive.")
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    ratio = np.power(x / half_saturation, slope)
    return ratio / (1.0 + ratio)

def add_media_features(frame: pd.DataFrame, alpha: float = .5) -> tuple[pd.DataFrame, dict[str, float]]:
    """Adds geo-causal adstock/Hill features using impression exposures."""
    out = frame.sort_values(["geo", "time"]).copy()
    half: dict[str, float] = {}
    for i in range(5):
        source, target = f"Channel{i}_impression", f"media_{i}"
        ad = out.groupby("geo", group_keys=False)[source].apply(
            lambda x: pd.Series(geometric_adstock(x.to_numpy(), alpha), index=x.index), include_groups=False
        )
        # groupby.apply returns an index-aligned Series on supported pandas versions
        out[f"adstock_{i}"] = ad.reindex(out.index).astype(float)
        h = float(np.quantile(out[f"adstock_{i}"], .5))
        half[target] = max(h, 1.0)
        out[target] = hill_saturation(out[f"adstock_{i}"].to_numpy(), half[target])
    return out, half
