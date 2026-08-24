"""Predictive power: information coefficients.

Rank IC is the headline metric. It is a Spearman correlation between a signal
at time t and the forward return over the next h events, so it is invariant to
monotone rescaling of the signal and far less hostage to the fat tails of
high-frequency returns than Pearson.

Nothing here is annualised, dressed up, or converted into a Sharpe. Turning a
correlation into money is the job of the backtest module, and keeping the two
apart is the point of the whole project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .significance import (
    circular_shift_pvalue,
    paired_block_bootstrap_ci,
    pearson,
    spearman,
)


def rank_ic(signal: pd.Series, forward_return: pd.Series) -> float:
    df = pd.concat([signal.rename("s"), forward_return.rename("r")], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    return spearman(df["s"].to_numpy(float), df["r"].to_numpy(float))


def ic_table(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    instrument: pd.Series,
    horizons: list[int],
    n_boot: int = 300,
    block: int = 500,
    seed: int = 0,
    permutation: bool = True,
    n_perm: int = 200,
) -> pd.DataFrame:
    """Rank IC for every signal by horizon cell, pooled across instruments.

    Confidence intervals come from a circular block bootstrap and the p-value
    from a block permutation, both of which respect the serial dependence that
    makes naive standard errors on high-frequency data untrustworthy.
    """
    rows = []
    for j, sig in enumerate(signals.columns):
        for k, h in enumerate(horizons):
            s, r = signals[sig], returns[f"fwd_ret_{h}"]
            df = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
            # Spearman is Pearson on ranks. Ranking once and resampling the
            # ranks gives the same point estimate at a fraction of the cost,
            # which is what makes a few hundred permutations affordable on a
            # full panel. It is a deliberate approximation inside the
            # resamples: ranks are not recomputed within each draw.
            sr = df["s"].rank()
            rr = df["r"].rank()
            point, lo, hi = paired_block_bootstrap_ci(
                sr, rr, pearson, block=block, n_boot=n_boot, seed=seed + 17 * j + k
            )
            # Exact over every admissible circular shift, so the p-value has
            # no sampling floor to trip the multiple-testing correction over.
            p, n_shifts = circular_shift_pvalue(s, r, min_shift=block) if permutation else (np.nan, 0)
            rows.append(
                {
                    "signal": sig,
                    "horizon": h,
                    "n_obs": len(df),
                    "rank_ic": point,
                    "ic_lo": lo,
                    "ic_hi": hi,
                    "pearson": pearson(df["s"].to_numpy(float), df["r"].to_numpy(float)) if len(df) > 30 else np.nan,
                    "p_value": p,
                    "p_value_floor": 1.0 / (n_shifts + 1) if permutation and n_shifts else np.nan,
                    "n_shifts": n_shifts,
                    "ic_significant": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0),
                }
            )
    return pd.DataFrame(rows)


def ic_by_instrument(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    instrument: pd.Series,
    horizons: list[int],
) -> pd.DataFrame:
    """The same matrix, split by instrument, to expose single-name results."""
    rows = []
    for name, idx in instrument.groupby(instrument, sort=True).groups.items():
        for sig in signals.columns:
            for h in horizons:
                rows.append(
                    {
                        "instrument": name,
                        "signal": sig,
                        "horizon": h,
                        "rank_ic": rank_ic(signals.loc[idx, sig], returns.loc[idx, f"fwd_ret_{h}"]),
                    }
                )
    return pd.DataFrame(rows)


def ic_cross_sectional(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    instrument: pd.Series,
    horizons: list[int],
    min_obs: int = 500,
) -> pd.DataFrame:
    """Per-instrument IC, aggregated with a cross-sectional standard error.

    Pooling every instrument into one Spearman is a weighted average dominated
    by whichever name has the most events, and it hides the case where one
    instrument carries the entire result. Treating each instrument as one
    observation of the IC gives an interpretable standard error and, more
    usefully, the count of instruments on which the sign actually holds.

    The t-statistic here is across instruments, not across time, so it answers
    "is this signal general" rather than "is this signal large".
    """
    rows = []
    per_instrument = ic_by_instrument(signals, returns, instrument, horizons)
    counts = instrument.value_counts()
    eligible = set(counts[counts >= min_obs].index)
    per_instrument = per_instrument[per_instrument["instrument"].isin(eligible)]

    for (sig, h), g in per_instrument.groupby(["signal", "horizon"], sort=True):
        ics = g["rank_ic"].dropna().to_numpy(float)
        n = len(ics)
        se = float(ics.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        mean = float(ics.mean()) if n else np.nan
        rows.append(
            {
                "signal": sig,
                "horizon": h,
                "n_instruments": n,
                "ic_mean": mean,
                "ic_median": float(np.median(ics)) if n else np.nan,
                "ic_std_across_instruments": float(ics.std(ddof=1)) if n > 1 else np.nan,
                "ic_stderr": se,
                "t_across_instruments": mean / se if se and np.isfinite(se) and se > 0 else np.nan,
                "share_same_sign_as_mean": float(np.mean(np.sign(ics) == np.sign(mean))) if n else np.nan,
                "ic_min": float(ics.min()) if n else np.nan,
                "ic_max": float(ics.max()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ic_matrix(ic_long: pd.DataFrame, value: str = "rank_ic") -> pd.DataFrame:
    """Long IC table to the signal by horizon matrix used in the paper."""
    return ic_long.pivot_table(index="signal", columns="horizon", values=value)
