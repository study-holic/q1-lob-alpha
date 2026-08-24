"""Paper figures.

Every figure is produced from a results table that has already been written to
disk, never from an in-memory object that only existed inside one run. If a
figure cannot be regenerated from the tables in ``results/tables``, it does not
go in the paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ic_decay(ic_long: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for sig, g in ic_long.sort_values("horizon").groupby("signal"):
        ax.plot(g["horizon"], g["rank_ic"], marker="o", label=sig)
        if {"ic_lo", "ic_hi"}.issubset(g.columns):
            ax.fill_between(g["horizon"], g["ic_lo"], g["ic_hi"], alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("horizon (events)")
    ax.set_ylabel("rank IC")
    ax.set_title("Signal decay: rank IC against forward horizon")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_cost_sensitivity(sweep: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["cost_multiple"], sweep["net_pnl"], marker="o", label="net P&L")
    ax.axhline(0, color="k", lw=0.8)
    be = sweep.attrs.get("breakeven_cost_multiple")
    if be is not None and np.isfinite(be):
        ax.axvline(be, color="crimson", ls="--", lw=1, label=f"breakeven at {be:.2f}x")
    ax.set_xlabel("transaction cost multiple of baseline assumption")
    ax.set_ylabel("net P&L (return units)")
    ax.set_title("Where the strategy stops being profitable")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_quantile_response(buckets: pd.DataFrame, signal: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(buckets["bucket"], 1e4 * buckets["mean_forward_return"],
           yerr=1e4 * buckets["stderr"], capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(f"{signal} quantile")
    ax.set_ylabel("mean forward return (bps)")
    ax.set_title(f"Forward return by {signal} quantile")
    return _save(fig, path)


def plot_equity_curve(pnl: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(pnl["timestamp"], pnl["gross"].cumsum(), label="gross", lw=1)
    ax.plot(pnl["timestamp"], pnl["net"].cumsum(), label="net of costs", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cumulative return units")
    ax.set_title("Gross against net cumulative P&L")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_regime_heatmap(ic_regime: pd.DataFrame, dimension: str, path: Path) -> Path:
    sub = ic_regime[ic_regime["regime_dimension"] == dimension]
    pivot = sub.pivot_table(index=["signal", "regime"], columns="horizon", values="rank_ic")
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(pivot) + 2))
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if pivot.size else 1.0
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{a} / {b}" for a, b in pivot.index])
    ax.set_xlabel("horizon (events)")
    ax.set_title(f"Rank IC by {dimension.replace('_', ' ')}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="rank IC")
    return _save(fig, path)


def plot_randomisation(rand: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rand["sharpe_gross"].dropna(), bins=12, alpha=0.75, label="block-shifted signal")
    real = rand.attrs.get("real_sharpe_gross")
    if real is not None and np.isfinite(real):
        ax.axvline(real, color="crimson", lw=1.5, label=f"actual signal ({real:.2f})")
    ax.set_xlabel("gross Sharpe (costs excluded, so turnover cannot confound)")
    ax.set_ylabel("draws")
    ax.set_title("Actual signal against its block-shifted null")
    ax.legend(fontsize=8)
    return _save(fig, path)
