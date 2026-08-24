"""Regime conditioning.

Regimes are assigned from *trailing* information only. Using full-sample
quantiles to label a regime is a look-ahead bug that is easy to miss, because
the labels feel like descriptions of the data rather than decisions made at a
point in time. Here each observation is bucketed against a rolling quantile of
its own past, so the label at time t could have been assigned at time t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.sessions import group_key
from ..orderbook.representation import depth, realised_volatility, relative_spread
from .ic import rank_ic

REGIME_LABELS = ["low", "medium", "high"]


def _rolling_bucket(x: pd.Series, groups: pd.Series, window: int, labels=REGIME_LABELS) -> pd.Series:
    """Label each point against the terciles of its own trailing window."""
    def label_one(s: pd.Series) -> pd.Series:
        minp = max(50, window // 10)
        lo = s.rolling(window, min_periods=minp).quantile(1 / 3).shift(1)
        hi = s.rolling(window, min_periods=minp).quantile(2 / 3).shift(1)
        out = pd.Series(index=s.index, dtype=object)
        out[s <= lo] = labels[0]
        out[(s > lo) & (s <= hi)] = labels[1]
        out[s > hi] = labels[2]
        out[lo.isna() | hi.isna()] = np.nan
        return out

    return x.groupby(groups, sort=False).transform(label_one)


def assign_regimes(book: pd.DataFrame, window: int = 2000, vol_window: int = 200) -> pd.DataFrame:
    """Return a frame of regime labels: spread, volatility and liquidity."""
    keys = group_key(book)
    return pd.DataFrame(
        {
            "spread_regime": _rolling_bucket(relative_spread(book), keys, window),
            "volatility_regime": _rolling_bucket(realised_volatility(book, vol_window, keys), keys, window),
            "liquidity_regime": _rolling_bucket(depth(book), keys, window),
        },
        index=book.index,
    )


def ic_by_regime(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    horizons: list[int],
    min_obs: int = 500,
) -> pd.DataFrame:
    """Rank IC inside every regime cell.

    Also reports the *spread* of IC across the buckets of a regime dimension,
    which is the quantity that tells you whether a signal is regime dependent
    or merely noisy.
    """
    rows = []
    for dim in regimes.columns:
        for bucket in REGIME_LABELS:
            mask = regimes[dim] == bucket
            n = int(mask.sum())
            for sig in signals.columns:
                for h in horizons:
                    rows.append(
                        {
                            "regime_dimension": dim,
                            "regime": bucket,
                            "signal": sig,
                            "horizon": h,
                            "n_obs": n,
                            "rank_ic": (
                                rank_ic(signals.loc[mask, sig], returns.loc[mask, f"fwd_ret_{h}"])
                                if n >= min_obs
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def regime_dependence(ic_regime: pd.DataFrame) -> pd.DataFrame:
    """Per signal, dimension and horizon: how much does IC move across buckets?"""
    g = ic_regime.groupby(["regime_dimension", "signal", "horizon"])["rank_ic"]
    out = g.agg(ic_min="min", ic_max="max", ic_mean="mean").reset_index()
    out["ic_range"] = out["ic_max"] - out["ic_min"]
    return out.sort_values("ic_range", ascending=False).reset_index(drop=True)
