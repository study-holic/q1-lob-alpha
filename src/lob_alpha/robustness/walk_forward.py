"""Walk forward validation.

Expanding window, never shuffled, with a purge gap between train and test so
that overlapping forward returns cannot straddle a boundary.

Splits are taken **within each instrument**, not on raw row position. A panel
is stored instrument-major, so slicing ``len(book)`` into equal blocks produces
folds whose test set is one ticker's entire trading day and whose training set
is a different ticker's. That is a cross-sectional holdout wearing a temporal
label: the purge gap purges 100 rows of an unrelated stock, and "out of sample"
means "a name the model has not seen" rather than "a period it has not seen".

    fold 1  per instrument:  train [====]  gap [-]  val [==]  test [==]
    fold 2  per instrument:  train [======]  gap [-]  val [==]  test [==]

Each fold's train, validation, and test masks therefore contain all five
instruments, and each block is a genuine time slice of every one of them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.engine import StrategyParams, run_backtest
from ..backtest.execution import ExecutionModel


def expanding_splits(
    n: int,
    n_folds: int = 5,
    min_train_fraction: float = 0.4,
    purge: int = 0,
) -> list[tuple[slice, slice, slice]]:
    """Row-index splits for a single contiguous series.

    Kept for single-instrument use and for the tests that check split geometry.
    On a panel, use :func:`grouped_splits`, which applies this geometry inside
    each instrument instead of across the concatenated frame.
    """
    start = int(n * min_train_fraction)
    step = max(1, (n - start) // n_folds)
    folds = []
    for k in range(n_folds):
        train_end = start + k * step
        val_end = min(n, train_end + step // 2)
        test_end = min(n, val_end + step - step // 2)
        if test_end - val_end < 50 or val_end - train_end < 50:
            break
        folds.append(
            (
                slice(0, max(0, train_end - purge)),
                slice(train_end, max(train_end, val_end - purge)),
                slice(val_end, test_end),
            )
        )
    return folds


def grouped_splits(
    groups: pd.Series,
    n_folds: int = 5,
    min_train_fraction: float = 0.4,
    purge: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Expanding-window masks applied inside each group, then unioned.

    Returns ``(train, validation, test)`` boolean masks over the whole frame.
    Every mask spans all groups, so a fold tests on a later period of each
    instrument rather than on a different instrument.
    """
    n = len(groups)
    train = [np.zeros(n, dtype=bool) for _ in range(n_folds)]
    val = [np.zeros(n, dtype=bool) for _ in range(n_folds)]
    test = [np.zeros(n, dtype=bool) for _ in range(n_folds)]
    used = [False] * n_folds

    positions = {g: np.flatnonzero((groups == g).to_numpy()) for g in groups.unique()}
    for idx in positions.values():
        for k, (tr, va, te) in enumerate(
            expanding_splits(len(idx), n_folds, min_train_fraction, purge)
        ):
            train[k][idx[tr]] = True
            val[k][idx[va]] = True
            test[k][idx[te]] = True
            used[k] = True
    return [(train[k], val[k], test[k]) for k in range(n_folds) if used[k]]


def walk_forward(
    book: pd.DataFrame,
    signals: pd.DataFrame,
    base_params: StrategyParams,
    cost_model: CostModel,
    execution: ExecutionModel,
    thresholds: Iterable[float] = (0.5, 1.0, 1.5, 2.0),
    candidate_signal_sets: list[list[str]] | None = None,
    n_folds: int = 5,
    purge: int = 101,
    regimes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Select on validation, evaluate once on test, per fold.

    Returns the per-fold table and a summary comparing the honest out-of-sample
    result against the in-sample optimum, which is the number most student
    projects accidentally report.
    """
    candidate_signal_sets = candidate_signal_sets or [base_params.signals]
    rows = []
    groups = book["instrument"] if "instrument" in book.columns else pd.Series(["all"] * len(book))
    for f, (tr, va, te) in enumerate(grouped_splits(groups, n_folds, purge=purge), start=1):
        best = None
        for sigset in candidate_signal_sets:
            for thr in thresholds:
                params = replace(base_params, signals=list(sigset), threshold=float(thr), weights=None)
                val = _slice_run(book, signals, params, cost_model, execution, regimes, va)
                score = val["metrics"]["sharpe_net"]
                if best is None or (np.isfinite(score) and score > best[0]):
                    best = (score, params)
        _, chosen = best
        test_run = _slice_run(book, signals, chosen, cost_model, execution, regimes, te)
        train_run = _slice_run(book, signals, chosen, cost_model, execution, regimes, tr)
        rows.append(
            {
                "fold": f,
                "chosen_signals": " + ".join(chosen.signals),
                "chosen_threshold": chosen.threshold,
                "train_sharpe": train_run["metrics"]["sharpe_net"],
                "validation_sharpe": best[0],
                "test_sharpe": test_run["metrics"]["sharpe_net"],
                "test_net_pnl": test_run["metrics"]["net_pnl"],
                "test_gross_pnl": test_run["metrics"]["gross_pnl"],
                "test_n_obs": test_run["metrics"]["n_obs"],
                "test_instruments": int(book.loc[te, "instrument"].nunique()) if "instrument" in book.columns else 1,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "mean_test_sharpe": float(table["test_sharpe"].mean()) if len(table) else np.nan,
        "mean_train_sharpe": float(table["train_sharpe"].mean()) if len(table) else np.nan,
        "folds_positive_test": int((table["test_sharpe"] > 0).sum()) if len(table) else 0,
        "n_folds": len(table),
        "in_sample_inflation": (
            float(table["train_sharpe"].mean() - table["test_sharpe"].mean()) if len(table) else np.nan
        ),
    }
    return table, summary


def _slice_run(book, signals, params, cost_model, execution, regimes, sl):
    """Run one fold. ``sl`` is a slice or a boolean mask over the frame."""
    b = book.loc[sl].reset_index(drop=True) if not isinstance(sl, slice) else book.iloc[sl].reset_index(drop=True)
    s = signals.loc[sl].reset_index(drop=True) if not isinstance(sl, slice) else signals.iloc[sl].reset_index(drop=True)
    if regimes is None:
        r = None
    else:
        r = regimes.loc[sl].reset_index(drop=True) if not isinstance(sl, slice) else regimes.iloc[sl].reset_index(drop=True)
    return run_backtest(b, s, params, cost_model, execution, r)
