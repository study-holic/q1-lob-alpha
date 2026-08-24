"""Cost and execution assumptions."""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.backtest.costs import CostModel
from lob_alpha.backtest.execution import ExecutionModel


def test_zero_cost_model_charges_nothing(small_book):
    model = CostModel(fixed_bps=0.0, half_spread_multiplier=0.0, impact_coefficient=0.0)
    assert (model.per_unit_turnover(small_book) == 0).all()


def test_cost_is_increasing_in_spread():
    book = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:01"]),
            "instrument": ["X", "X"],
            "bid_price": [99.99, 99.90],
            "bid_size": [100.0, 100.0],
            "ask_price": [100.01, 100.10],
            "ask_size": [100.0, 100.0],
        }
    )
    c = CostModel().per_unit_turnover(book)
    assert c.iloc[1] > c.iloc[0]


def test_rescaling_is_linear(small_book):
    base = CostModel(fixed_bps=1.0).per_unit_turnover(small_book)
    doubled = CostModel(fixed_bps=1.0).rescaled(2.0).per_unit_turnover(small_book)
    np.testing.assert_allclose(doubled.to_numpy(), 2 * base.to_numpy())


def test_execution_latency_cannot_be_zero():
    with pytest.raises(ValueError):
        ExecutionModel(level=2, latency_events=0)


def test_queue_aware_level_is_declared_not_faked():
    with pytest.raises(NotImplementedError):
        ExecutionModel(level=5)


def test_latency_shifts_the_position():
    target = pd.Series([0.0, 1.0, 1.0, -1.0])
    inst = pd.Series(["X"] * 4)
    held = ExecutionModel(level=2, latency_events=1).realise(target, inst)
    assert list(held) == [0.0, 0.0, 1.0, 1.0]


def test_partial_fills_approach_the_target_gradually():
    target = pd.Series([1.0] * 6)
    inst = pd.Series(["X"] * 6)
    held = ExecutionModel(level=4, fill_ratio=0.5, latency_events=1).realise(target, inst)
    assert held.iloc[-1] < 1.0
    assert held.is_monotonic_increasing


def test_mid_fill_level_has_no_cost_multiplier():
    assert ExecutionModel(level=1).cost_multiplier() == 0.0
    assert ExecutionModel(level=3, slippage_fraction_of_spread=0.25).cost_multiplier() > 1.0
