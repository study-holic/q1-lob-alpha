"""The backtest engine.

The strategy is deliberately close to trivial: standardise the signal against
its own trailing distribution, go long above a threshold, short below the
negative threshold, flat in between, subject to a position limit. If a simple
rule on a validated signal makes money, the reason is legible. If an elaborate
rule makes money, the reason is usually the elaboration.

Two rules are enforced structurally rather than by good intentions:

* standardisation uses trailing statistics only, so the z-score at time t is
  computable at time t;
* the execution model shifts the position by at least one event, so a signal
  observed at t is traded on the book at t + 1 at the earliest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.sessions import group_key
from ..orderbook.representation import mid as _mid
from .costs import CostModel
from .execution import ExecutionModel

SECONDS_PER_YEAR = 252 * 6.5 * 3600


@dataclass
class StrategyParams:
    signals: list[str] = field(default_factory=lambda: ["ofi"])
    weights: list[float] | None = None
    threshold: float = 1.0
    zscore_window: int = 2000
    position_limit: float = 1.0
    regime_filter: tuple[str, list[str]] | None = None  # e.g. ("spread_regime", ["high"])


def trailing_zscore(x: pd.Series, groups: pd.Series, window: int) -> pd.Series:
    minp = max(50, window // 10)
    def z(s: pd.Series) -> pd.Series:
        mu = s.rolling(window, min_periods=minp).mean()
        sd = s.rolling(window, min_periods=minp).std()
        return (s - mu) / sd.replace(0.0, np.nan)
    return x.groupby(groups, sort=False).transform(z)


def build_position(
    signals: pd.DataFrame,
    groups: pd.Series,
    params: StrategyParams,
    regimes: pd.DataFrame | None = None,
) -> pd.Series:
    """Target position in [-limit, limit], computable at time t."""
    cols = params.signals
    weights = params.weights or [1.0] * len(cols)
    if len(weights) != len(cols):
        raise ValueError("weights and signals must be the same length")

    z = sum(
        w * trailing_zscore(signals[c], groups, params.zscore_window)
        for c, w in zip(cols, weights)
    ) / np.sqrt(sum(w * w for w in weights))

    pos = pd.Series(0.0, index=signals.index)
    pos[z > params.threshold] = params.position_limit
    pos[z < -params.threshold] = -params.position_limit
    pos[z.isna()] = 0.0

    if params.regime_filter is not None and regimes is not None:
        dim, allowed = params.regime_filter
        pos = pos.where(regimes[dim].isin(allowed), 0.0)
    return pos


def run_backtest(
    book: pd.DataFrame,
    signals: pd.DataFrame,
    params: StrategyParams,
    cost_model: CostModel,
    execution: ExecutionModel,
    regimes: pd.DataFrame | None = None,
) -> dict:
    """Run one strategy and return ``{'metrics', 'pnl', 'position'}``."""
    instrument = book["instrument"]
    groups = group_key(book)
    target = build_position(signals, groups, params, regimes)
    position = execution.realise(target, groups)

    m = _mid(book)
    # Return from t to t+1, within session. The last event of a session has no
    # next event, so the position held into the close earns nothing rather than
    # earning the overnight gap.
    ret = m.groupby(groups, sort=False).pct_change().shift(-1)
    ret = ret.where(groups == groups.shift(-1))

    turnover = position.groupby(groups, sort=False).diff().abs().fillna(position.abs())
    unit_cost = cost_model.per_unit_turnover(book) * execution.cost_multiplier()

    gross = (position * ret).fillna(0.0)
    cost = (turnover * unit_cost).fillna(0.0)
    net = gross - cost

    pnl = pd.DataFrame(
        {
            "timestamp": book["timestamp"],
            "instrument": instrument,
            "position": position,
            "turnover": turnover,
            "mid_return": ret,
            "gross": gross,
            "cost": cost,
            "net": net,
        }
    )
    metrics = compute_metrics(pnl, book)
    metrics["execution"] = execution.describe()
    metrics["threshold"] = params.threshold
    metrics["signals"] = " + ".join(params.signals)
    return {"metrics": metrics, "pnl": pnl, "position": position, "target": target}


def compute_metrics(pnl: pd.DataFrame, book: pd.DataFrame) -> dict:
    net, gross = pnl["net"], pnl["gross"]
    traded = pnl["turnover"] > 0
    n_trades = int(traded.sum())
    # Grouped: an ungrouped diff spans instrument boundaries and produces
    # large negative gaps where the frame moves from one name to the next.
    keys = group_key(book)
    dt = pnl["timestamp"].groupby(keys, sort=False).diff().dt.total_seconds()
    median_dt = float(np.nanmedian(dt)) if len(dt.dropna()) else np.nan
    ann = np.sqrt(SECONDS_PER_YEAR / median_dt) if median_dt and median_dt > 0 else np.nan

    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    active = pnl.loc[pnl["position"] != 0, "net"]
    held = pnl.loc[pnl["position"] != 0]
    moved = held.loc[held["mid_return"].fillna(0) != 0, "gross"]

    return {
        "n_obs": len(pnl),
        "gross_pnl": float(gross.sum()),
        "net_pnl": float(net.sum()),
        "total_cost": float(pnl["cost"].sum()),
        "cost_share_of_gross": float(pnl["cost"].sum() / gross.sum()) if gross.sum() != 0 else np.nan,
        # Per-event Sharpe is the primary figure. The annualised version is
        # reported alongside it because that is what people expect to see, but
        # sqrt-of-time scaling from a sub-millisecond event assumes independence across
        # events, which is exactly the assumption the block bootstrap exists to
        # doubt. Compare strategies on the per-event number.
        "sharpe_net": float(net.mean() / net.std()) if net.std() > 0 else np.nan,
        "sharpe_gross": float(gross.mean() / gross.std()) if gross.std() > 0 else np.nan,
        "sharpe_net_annualised": float(net.mean() / net.std() * ann) if net.std() > 0 else np.nan,
        "sharpe_gross_annualised": float(gross.mean() / gross.std() * ann) if gross.std() > 0 else np.nan,
        "annualisation_factor": float(ann) if np.isfinite(ann) else np.nan,
        # Reported two ways. The unconditional rate has a denominator in which
        # 87% of events carry a mid return of exactly zero, because a size-only
        # book update changes the state without moving the price. That is not a
        # wrong trade; it is no trade outcome at all. The conditional rate is
        # the one that says whether the signal points the right way.
        "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "hit_rate_conditional": (
            float((moved > 0).mean()) if len(moved) else np.nan
        ),
        # Share of events at which the mid did not move at all. Computed from
        # the return, not from P&L: a flat position also produces zero P&L, and
        # conflating the two would overstate this.
        "zero_return_share": float((pnl["mid_return"].fillna(0) == 0).mean()),
        "n_directional_events": int(len(moved)),
        "median_event_spacing_s": median_dt,
        "avg_trade_net": float(net.sum() / n_trades) if n_trades else np.nan,
        "n_trades": n_trades,
        "turnover_total": float(pnl["turnover"].sum()),
        "exposure": float((pnl["position"] != 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "breakeven_cost_multiple": _breakeven_multiple(gross.sum(), pnl["cost"].sum()),
    }


def _breakeven_multiple(gross: float, cost: float) -> float:
    """How many times the assumed cost the strategy could pay before losing."""
    if cost <= 0:
        return np.inf if gross > 0 else np.nan
    return float(gross / cost)


def adverse_selection(
    book: pd.DataFrame,
    pnl: pd.DataFrame,
    horizons: list[int],
    conditioner: pd.Series | None = None,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Mid price drift after a fill, signed by trade direction.

        AS(h) = sign * (m_{t+h} - m_t)/m_t

    Negative values mean the market moved against the fill, which is the
    signature of being picked off. A strategy can be profitable in the backtest
    and still be losing this measurement, which usually means the profit lives
    in an execution assumption rather than in the signal.
    """
    instrument = book["instrument"]
    m = _mid(book)
    groups = group_key(book)
    trades = pnl["position"].groupby(groups, sort=False).diff().fillna(pnl["position"])
    side = np.sign(trades)
    mask = side != 0
    if not mask.any():
        return pd.DataFrame()

    rows = {}
    for h in horizons:
        fwd = m.groupby(groups, sort=False).shift(-h)
        rows[f"as_{h}"] = side * (fwd - m) / m
    out = pd.DataFrame(rows)[mask.to_numpy()]

    if conditioner is None:
        summary = out.mean().rename("mean_adverse_selection").to_frame().reset_index()
        summary.columns = ["horizon_column", "mean_adverse_selection"]
        summary["n_fills"] = len(out)
        return summary

    c = conditioner[mask.to_numpy()]
    # Ranked within instrument. On a mixed panel the absolute spread is an
    # instrument label: AAPL's median spread is fifteen times INTC's, so a
    # pooled quintile sorts names, not market states, and any "wide spread"
    # conclusion is really a statement about which tickers are expensive.
    inst = instrument[mask.to_numpy()]
    ranked = c.groupby(inst, sort=False).rank(pct=True, method="first")
    bucket = pd.qcut(ranked, n_buckets, labels=False, duplicates="drop") + 1
    out = out.assign(bucket=bucket.to_numpy())
    summary = out.groupby("bucket").mean().reset_index()
    summary["n_fills"] = out.groupby("bucket").size().to_numpy()
    return summary
