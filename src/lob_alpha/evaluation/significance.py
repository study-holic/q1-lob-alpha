"""Statistical significance under time dependence.

Overlapping forward returns and autocorrelated signals break the iid
assumption behind ordinary standard errors, usually in the direction that
flatters the result. Everything here is either a block method or a permutation
that preserves the dependence structure it is trying to test against.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index draw."""
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def block_bootstrap_ci(
    x: np.ndarray | pd.Series,
    stat: Callable[[np.ndarray], float],
    block: int = 500,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, lower, upper)`` for a statistic of one series."""
    a = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(a) < 10:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(stat(a))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = stat(a[block_bootstrap_indices(len(a), block, rng)])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_block_bootstrap_ci(
    x: pd.Series,
    y: pd.Series,
    stat: Callable[[np.ndarray, np.ndarray], float],
    block: int = 500,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Block bootstrap for a statistic of two aligned series, e.g. rank IC."""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 30:
        return float("nan"), float("nan"), float("nan")
    xv, yv = df["x"].to_numpy(float), df["y"].to_numpy(float)
    rng = np.random.default_rng(seed)
    point = float(stat(xv, yv))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = block_bootstrap_indices(len(df), block, rng)
        draws[b] = stat(xv[idx], yv[idx])
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def circular_shift_pvalue(
    x: pd.Series,
    y: pd.Series,
    min_shift: int = 500,
    use_ranks: bool = True,
) -> tuple[float, int]:
    """Exact circular-shift p-value for a correlation, via one FFT.

    The permutation null here is "shift the signal so far that its alignment
    with future returns is destroyed, while its own autocorrelation survives".
    Drawing a few hundred random shifts estimates that null; a circular
    cross-correlation evaluates it at *every* admissible shift at once, in
    O(n log n) rather than O(n_perm * n).

    Two consequences beyond speed. The p-value is exact rather than sampled, so
    it carries no ``1/(n_perm+1)`` resolution floor, which matters when the
    result must clear a multiple-testing correction over a large family. And
    the test stops depending on a seed.

    Returns ``(p_value, n_shifts_considered)``.
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(df)
    if n < 100 or n <= 4 * min_shift:
        return float("nan"), 0
    a = df["x"].rank().to_numpy(float) if use_ranks else df["x"].to_numpy(float)
    b = df["y"].rank().to_numpy(float) if use_ranks else df["y"].to_numpy(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    if denom <= 0:
        return float("nan"), 0

    # cross[s] = sum_t a[(t + s) mod n] * b[t], for every shift s.
    cross = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n)
    corr = cross / denom
    observed = abs(corr[0])

    shifts = np.arange(n)
    admissible = (shifts >= min_shift) & (shifts <= n - min_shift)
    null = np.abs(corr[admissible])
    if null.size == 0:
        return float("nan"), 0
    return float((null >= observed).sum() + 1) / (null.size + 1), int(null.size)


def block_permutation_pvalue(
    x: pd.Series,
    y: pd.Series,
    stat: Callable[[np.ndarray, np.ndarray], float],
    block: int = 500,
    n_perm: int = 300,
    seed: int = 0,
    exact: bool = True,
) -> float:
    """Two-sided p-value from circularly block-shifting the signal.

    Defaults to the exact FFT evaluation above. Set ``exact=False`` to fall
    back to sampling ``n_perm`` random shifts, which is what a statistic other
    than a correlation requires.
    """
    if exact:
        p, _ = circular_shift_pvalue(x, y, min_shift=block)
        if np.isfinite(p):
            return p
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    xv, yv = df["x"].to_numpy(float), df["y"].to_numpy(float)
    observed = abs(stat(xv, yv))
    rng = np.random.default_rng(seed)
    n = len(xv)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(block, max(block + 1, n - block)))
        if abs(stat(np.roll(xv, shift), yv)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.05, m: int | None = None) -> pd.DataFrame:
    """Benjamini and Hochberg step-up, alongside Bonferroni for reference.

    ``m`` is the family size. It defaults to the number of p-values supplied,
    but when only part of a family is being corrected the caller should pass
    the full count, which makes both procedures stricter.
    """
    p = pvalues.dropna().sort_values()
    if len(p) == 0:
        return pd.DataFrame(columns=["p_value", "bh_threshold", "bh_reject", "bonferroni_reject"])
    m = int(m or len(p))
    ranks = np.arange(1, len(p) + 1)
    thresh = alpha * ranks / m
    below = p.to_numpy() <= thresh
    k = np.max(np.flatnonzero(below)) + 1 if below.any() else 0
    reject = np.zeros(len(p), dtype=bool)
    reject[:k] = True
    return pd.DataFrame(
        {
            "p_value": p,
            "bh_threshold": thresh,
            "bh_reject": reject,
            "bonferroni_reject": p.to_numpy() <= alpha / m,
        }
    )


def newey_west_tstat(y: np.ndarray, lags: int | None = None) -> float:
    """t-statistic for a zero mean, with a Newey and West correction."""
    a = np.asarray(pd.Series(y).dropna(), dtype=float)
    n = len(a)
    if n < 10:
        return float("nan")
    lags = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = a - a.mean()
    var = float(e @ e) / n
    for l in range(1, lags + 1):
        cov = float(e[l:] @ e[:-l]) / n
        var += 2 * (1 - l / (lags + 1)) * cov
    if var <= 0:
        return float("nan")
    return float(a.mean() / np.sqrt(var / n))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
