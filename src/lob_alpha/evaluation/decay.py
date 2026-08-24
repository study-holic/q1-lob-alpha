"""Signal decay.

The IC by horizon curve answers the question a trading desk actually asks:
not "is there information" but "for how long is it there". A signal with a
smaller peak IC that persists for fifty events can be far more tradable than a
larger one that is gone by the second event, because the second one has to pay
the spread over a shorter window.

The half-life reported here is the horizon at which |IC| first falls below half
its peak, interpolated linearly between the bracketing horizons. It is a
descriptive statistic, not a fitted decay model, and it is reported in both
events and seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decay_curve(ic_long: pd.DataFrame) -> pd.DataFrame:
    """IC against horizon, one row per signal by horizon, sorted for plotting."""
    cols = [c for c in ["signal", "horizon", "rank_ic", "ic_lo", "ic_hi"] if c in ic_long.columns]
    return ic_long[cols].sort_values(["signal", "horizon"]).reset_index(drop=True)


def half_life(ic_long: pd.DataFrame, clock: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for sig, g in ic_long.sort_values("horizon").groupby("signal", sort=True):
        h = g["horizon"].to_numpy(float)
        ic = np.abs(g["rank_ic"].to_numpy(float))
        if not np.isfinite(ic).any():
            rows.append({"signal": sig, "peak_horizon": np.nan, "peak_abs_ic": np.nan, "half_life_events": np.nan})
            continue
        i_peak = int(np.nanargmax(ic))
        peak = ic[i_peak]
        target = peak / 2.0
        hl = np.nan
        for i in range(i_peak + 1, len(ic)):
            if np.isfinite(ic[i]) and ic[i] <= target:
                x0, x1 = h[i - 1], h[i]
                y0, y1 = ic[i - 1], ic[i]
                hl = x0 if y0 == y1 else x0 + (y0 - target) * (x1 - x0) / (y0 - y1)
                break
        rows.append(
            {
                "signal": sig,
                "peak_horizon": h[i_peak],
                "peak_abs_ic": peak,
                "half_life_events": hl,
                "decayed_within_tested_range": bool(np.isfinite(hl)),
            }
        )
    out = pd.DataFrame(rows)
    if clock is not None and not clock.empty:
        sec_per_event = clock["median_seconds"].to_numpy() / clock["horizon_events"].to_numpy()
        rate = float(np.nanmedian(sec_per_event))
        out["half_life_seconds"] = out["half_life_events"] * rate
    return out
