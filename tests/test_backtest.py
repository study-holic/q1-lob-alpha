"""Backtest accounting identities."""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.backtest.costs import CostModel
from lob_alpha.backtest.engine import StrategyParams, build_position, run_backtest, trailing_zscore
from lob_alpha.backtest.execution import ExecutionModel
from lob_alpha.signals import build_signals

PARAMS = StrategyParams(signals=["ofi"], threshold=0.5, zscore_window=500)


@pytest.fixture(scope="module")
def signals(small_book):
    return build_signals(small_book, ["ofi", "queue_imbalance", "microprice_deviation", "spread"])


def test_zero_cost_means_net_equals_gross(small_book, signals):
    free = CostModel(fixed_bps=0.0, half_spread_multiplier=0.0, impact_coefficient=0.0)
    r = run_backtest(small_book, signals, PARAMS, free, ExecutionModel(level=1))
    np.testing.assert_allclose(r["pnl"]["net"].to_numpy(), r["pnl"]["gross"].to_numpy())
    assert r["metrics"]["net_pnl"] == pytest.approx(r["metrics"]["gross_pnl"])


def test_costs_can_only_reduce_pnl(small_book, signals):
    r = run_backtest(small_book, signals, PARAMS, CostModel(), ExecutionModel(level=2))
    assert r["metrics"]["net_pnl"] <= r["metrics"]["gross_pnl"]
    assert (r["pnl"]["cost"] >= 0).all()


def test_flipping_the_signal_flips_the_pnl(small_book, signals):
    free = CostModel(fixed_bps=0.0, half_spread_multiplier=0.0, impact_coefficient=0.0)
    a = run_backtest(small_book, signals, PARAMS, free, ExecutionModel(level=1))
    b = run_backtest(small_book, -signals, PARAMS, free, ExecutionModel(level=1))
    assert a["metrics"]["gross_pnl"] == pytest.approx(-b["metrics"]["gross_pnl"], abs=1e-12)


def test_position_respects_the_limit_and_the_threshold(small_book, signals):
    params = StrategyParams(signals=["ofi"], threshold=1.0, zscore_window=500, position_limit=0.5)
    pos = build_position(signals, small_book["instrument"], params)
    assert pos.abs().max() <= 0.5
    assert set(pos.unique()).issubset({-0.5, 0.0, 0.5})


def test_position_is_computable_at_time_t(small_book, signals):
    """Truncating the future must not change any past position."""
    params = StrategyParams(signals=["ofi"], threshold=1.0, zscore_window=500)
    full = build_position(signals, small_book["instrument"], params)
    cut = len(small_book) // 2
    truncated = build_position(
        signals.iloc[:cut], small_book["instrument"].iloc[:cut], params
    )
    pd.testing.assert_series_equal(full.iloc[:cut], truncated, check_names=False)


def test_no_trading_means_no_pnl(small_book, signals):
    params = StrategyParams(signals=["ofi"], threshold=1e9, zscore_window=500)
    r = run_backtest(small_book, signals, params, CostModel(), ExecutionModel(level=2))
    assert r["metrics"]["net_pnl"] == pytest.approx(0.0)
    assert r["metrics"]["n_trades"] == 0


def test_turnover_matches_position_changes(small_book, signals):
    r = run_backtest(small_book, signals, PARAMS, CostModel(), ExecutionModel(level=2))
    pos = r["pnl"]["position"]
    expected = pos.diff().abs().fillna(pos.abs()).sum()
    assert r["metrics"]["turnover_total"] == pytest.approx(expected)
