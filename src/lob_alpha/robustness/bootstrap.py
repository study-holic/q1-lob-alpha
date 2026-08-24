"""Uncertainty around performance statistics.

A Sharpe ratio reported without an interval is a point estimate of a noisy
quantity dressed up as a fact. Everything here uses block resampling so the
serial dependence in a high-frequency P&L series survives the resample, and
the P&L attribution answers the question that kills more strategies than any
other: was this one exceptional afternoon?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.significance import block_bootstrap_ci, newey_west_tstat


def sharpe(x: np.ndarray) -> float:
    sd = np.std(x)
    return float(np.mean(x) / sd) if sd > 0 else float("nan")


def sharpe_ci(pnl: pd.Series, block: int = 500, n_boot: int = 500, seed: int = 0, annualisation: float = 1.0):
    point, lo, hi = block_bootstrap_ci(pnl, sharpe, block=block, n_boot=n_boot, seed=seed)
    return {
        "sharpe": point * annualisation,
        "sharpe_lo": lo * annualisation,
        "sharpe_hi": hi * annualisation,
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0),
        "newey_west_t": newey_west_tstat(pnl.dropna().to_numpy(float)),
    }


def pnl_attribution(pnl: pd.DataFrame, by: str = "day") -> pd.DataFrame:
    """Break net P&L down by period, instrument or regime bucket."""
    df = pnl.copy()
    if by == "day":
        key = df["timestamp"].dt.date
    elif by == "week":
        key = df["timestamp"].dt.to_period("W").astype(str)
    elif by in df.columns:
        key = df[by]
    else:
        raise ValueError(f"cannot attribute by {by!r}")
    out = df.groupby(key)["net"].agg(net_pnl="sum", n_obs="size").reset_index(names=by)
    total = out["net_pnl"].sum()
    out["share_of_total"] = out["net_pnl"] / total if total != 0 else np.nan
    return out.sort_values("net_pnl", ascending=False).reset_index(drop=True)


def concentration(attribution: pd.DataFrame) -> dict:
    """How much of the total sits in the single best period?"""
    if attribution.empty:
        return {}
    net = attribution["net_pnl"]
    total = float(net.sum())
    return {
        "n_periods": len(net),
        "positive_periods": int((net > 0).sum()),
        "best_period_share": float(net.max() / total) if total != 0 else np.nan,
        "top_5pct_share": float(net.nlargest(max(1, len(net) // 20)).sum() / total) if total != 0 else np.nan,
        "total_without_best_period": float(total - net.max()),
    }
