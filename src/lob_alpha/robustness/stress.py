"""Can I make the alpha disappear?

Each function below is an attempt to destroy the headline result. A result
that survives all of them is worth writing up. A result that survives none of
them is also worth writing up, which is the part most projects skip.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.engine import StrategyParams, run_backtest
from ..backtest.execution import ExecutionModel


def cost_sensitivity(
    book: pd.DataFrame,
    signals: pd.DataFrame,
    params: StrategyParams,
    cost_model: CostModel,
    execution: ExecutionModel,
    multiples=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
    regimes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sweep the cost assumption and find where profitability flips."""
    rows = []
    for k in multiples:
        run = run_backtest(book, signals, params, cost_model.rescaled(k), execution, regimes)
        m = run["metrics"]
        rows.append(
            {
                "cost_multiple": k,
                "gross_pnl": m["gross_pnl"],
                "net_pnl": m["net_pnl"],
                "sharpe_net": m["sharpe_net"],
                "cost_share_of_gross": m["cost_share_of_gross"],
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["breakeven_cost_multiple"] = _breakeven(out)
    return out


def _breakeven(sweep: pd.DataFrame) -> float:
    """Linear interpolation of the cost multiple where net P&L crosses zero."""
    s = sweep.sort_values("cost_multiple")
    x, y = s["cost_multiple"].to_numpy(float), s["net_pnl"].to_numpy(float)
    for i in range(1, len(x)):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and y[i - 1] > 0 >= y[i]:
            return float(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
    if (y > 0).all():
        return float("inf")  # profitable across the whole sweep
    # Never crosses because it never starts positive. Returning 0.0 here would
    # read as "breaks even at zero cost", which is the opposite of the truth:
    # the strategy loses money even when trading is free.
    return float("nan")


def threshold_sensitivity(
    book, signals, params, cost_model, execution, thresholds=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0), regimes=None
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        run = run_backtest(book, signals, replace(params, threshold=float(thr)), cost_model, execution, regimes)
        m = run["metrics"]
        rows.append(
            {
                "threshold": thr,
                "net_pnl": m["net_pnl"],
                "sharpe_net": m["sharpe_net"],
                "n_trades": m["n_trades"],
                "exposure": m["exposure"],
            }
        )
    return pd.DataFrame(rows)


def execution_ladder(book, signals, params, cost_model, levels=(1, 2, 3, 4), regimes=None) -> pd.DataFrame:
    rows = []
    for level in levels:
        ex = ExecutionModel(level=level)
        run = run_backtest(book, signals, params, cost_model, ex, regimes)
        m = run["metrics"]
        rows.append(
            {
                "execution_level": level,
                "description": ex.describe(),
                "gross_pnl": m["gross_pnl"],
                "net_pnl": m["net_pnl"],
                "sharpe_net": m["sharpe_net"],
            }
        )
    return pd.DataFrame(rows)


def instrument_stability(book, signals, params, cost_model, execution, regimes=None) -> pd.DataFrame:
    """Does the result live in one name?"""
    rows = []
    for name, idx in book.groupby("instrument", sort=True).groups.items():
        b = book.loc[idx].reset_index(drop=True)
        s = signals.loc[idx].reset_index(drop=True)
        r = regimes.loc[idx].reset_index(drop=True) if regimes is not None else None
        m = run_backtest(b, s, params, cost_model, execution, r)["metrics"]
        rows.append(
            {
                "instrument": name,
                "net_pnl": m["net_pnl"],
                "sharpe_net": m["sharpe_net"],
                "n_trades": m["n_trades"],
            }
        )
    return pd.DataFrame(rows)


def period_stability(book, signals, params, cost_model, execution, n_periods: int = 5, regimes=None) -> pd.DataFrame:
    """Does the result live in one stretch of time?

    Split by clock time, not by row position. A panel is stored
    instrument-major, so slicing the frame into equal row blocks produces
    groups of instruments wearing the label "period", which is a different
    test wearing the same name.
    """
    ts = book["timestamp"]
    edges = pd.to_datetime(
        np.linspace(ts.min().value, ts.max().value, n_periods + 1)
    )
    rows = []
    for i in range(n_periods):
        lo, hi = edges[i], edges[i + 1]
        mask = (ts >= lo) & (ts < hi) if i < n_periods - 1 else (ts >= lo) & (ts <= hi)
        if mask.sum() < 100:
            continue
        b = book.loc[mask].reset_index(drop=True)
        s = signals.loc[mask.to_numpy()].reset_index(drop=True)
        r = regimes.loc[mask.to_numpy()].reset_index(drop=True) if regimes is not None else None
        m = run_backtest(b, s, params, cost_model, execution, r)["metrics"]
        rows.append(
            {
                "period": i + 1,
                "start": lo,
                "end": hi,
                "n_obs": int(mask.sum()),
                "instruments": b["instrument"].nunique(),
                "gross_pnl": m["gross_pnl"],
                "net_pnl": m["net_pnl"],
                "sharpe_net": m["sharpe_net"],
                "sharpe_gross": m["sharpe_gross"],
            }
        )
    return pd.DataFrame(rows)


def signal_randomisation(
    book, signals, params, cost_model, execution, n_draws: int = 20, block: int = 500, seed: int = 0, regimes=None
) -> pd.DataFrame:
    """Block-shift the signal and re-run. This is the null the result must beat.

    Shifting in blocks preserves the signal's own autocorrelation, so what is
    destroyed is only its alignment with future returns. If a shifted signal
    trades about as well as the real one, the strategy was harvesting something
    structural, not predictive.
    """
    rng = np.random.default_rng(seed)
    real = run_backtest(book, signals, params, cost_model, execution, regimes)["metrics"]
    rows = []
    n = len(book)
    for d in range(n_draws):
        shift = int(rng.integers(block, max(block + 1, n - block)))
        shuffled = signals.apply(lambda col: pd.Series(np.roll(col.to_numpy(), shift), index=col.index))
        m = run_backtest(book, shuffled, params, cost_model, execution, regimes)["metrics"]
        rows.append({
            "draw": d + 1, "shift": shift,
            "net_pnl": m["net_pnl"], "sharpe_net": m["sharpe_net"],
            "gross_pnl": m["gross_pnl"], "sharpe_gross": m["sharpe_gross"],
        })
    out = pd.DataFrame(rows)
    out.attrs["real_net_pnl"] = real["net_pnl"]
    out.attrs["real_sharpe"] = real["sharpe_net"]
    out.attrs["real_gross_pnl"] = real["gross_pnl"]
    out.attrs["real_sharpe_gross"] = real["sharpe_gross"]

    # The verdict is taken on GROSS Sharpe. Judging it on net confounds
    # predictive content with turnover: a shifted signal trades at different
    # moments and therefore pays a different bill, so a strategy can "beat its
    # null" purely by trading when the spread happens to be narrow. Gross
    # isolates the only thing the shift is supposed to destroy, which is the
    # alignment between signal and future return.
    for label, col, real_value in (
        ("", "sharpe_gross", real["sharpe_gross"]),
        ("_net", "sharpe_net", real["sharpe_net"]),
    ):
        finite = out[col].dropna()
        out.attrs[f"empirical_pvalue{label}"] = (
            float((finite >= real_value).sum() + 1) / (len(finite) + 1) if len(finite) else np.nan
        )
    return out


def subsampling_sensitivity(
    book, signals, params, cost_model, execution, strides=(1, 2, 5), regimes=None
) -> pd.DataFrame:
    """Does the result survive looking at the book less often?"""
    rows = []
    for stride in strides:
        b = book.iloc[::stride].reset_index(drop=True)
        s = signals.iloc[::stride].reset_index(drop=True)
        r = regimes.iloc[::stride].reset_index(drop=True) if regimes is not None else None
        m = run_backtest(b, s, params, cost_model, execution, r)["metrics"]
        rows.append({"stride": stride, "n_obs": m["n_obs"], "net_pnl": m["net_pnl"], "sharpe_net": m["sharpe_net"]})
    return pd.DataFrame(rows)
