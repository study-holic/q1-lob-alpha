"""Does the research machinery find what is there, and nothing when nothing is?

These are the tests that make the rest of the results believable. If the
pipeline cannot recover a planted relationship, a null result means nothing. If
it finds a relationship in a market built to have none, a positive result means
nothing either.
"""

import numpy as np
import pytest

from lob_alpha.backtest.costs import CostModel
from lob_alpha.backtest.engine import StrategyParams, run_backtest
from lob_alpha.backtest.execution import ExecutionModel
from lob_alpha.data.synthetic import simulate_book
from lob_alpha.evaluation.ic import rank_ic
from lob_alpha.evaluation.regimes import assign_regimes, ic_by_regime
from lob_alpha.evaluation.significance import block_permutation_pvalue, spearman
from lob_alpha.signals import build_signals
from lob_alpha.targets.forward_returns import forward_returns

SIGNAL_SET = ["ofi", "queue_imbalance", "microprice_deviation", "spread"]


def _ofi_ic(book, horizon=10):
    sig = build_signals(book, ["ofi"])["ofi"]
    fwd = forward_returns(book, [horizon])[f"fwd_ret_{horizon}"]
    return rank_ic(sig, fwd)


def test_planted_ofi_signal_is_recovered():
    book = simulate_book(n_events=30000, mode="ofi", beta=0.35, seed=11)
    assert _ofi_ic(book) > 0.02


def test_stronger_planting_gives_a_stronger_ic():
    weak = _ofi_ic(simulate_book(n_events=30000, mode="ofi", beta=0.10, seed=12))
    strong = _ofi_ic(simulate_book(n_events=30000, mode="ofi", beta=0.45, seed=12))
    assert strong > weak


def test_null_market_ofi_ic_is_centred_on_zero():
    """Across independent null realisations the IC must average to nothing.

    A single realisation is the wrong unit of evidence here. With a 20-event
    rolling signal and 10-event overlapping returns the effective sample is two
    orders of magnitude smaller than the row count, so any one seed lands a few
    hundredths away from zero. What has to hold is that the *distribution* is
    centred, which is what this checks.
    """
    ics = np.array([
        _ofi_ic(simulate_book(n_events=20000, mode="null", seed=100 + s))
        for s in range(6)
    ])
    assert abs(ics.mean()) < 0.02
    assert np.abs(ics).max() < 0.08


def test_null_market_permutation_test_is_not_wildly_miscalibrated():
    """The shifted null must reject rarely on data with nothing in it.

    It is mildly anti-conservative (see docs/protocol.md): a circular shift
    preserves each series' own autocorrelation but not the variation across
    independent realisations, so its null is roughly 40% narrower than the
    across-seed spread. The bar here is that it is not badly broken.
    """
    rejects = 0
    for seed in range(5):
        book = simulate_book(n_events=20000, mode="null", seed=200 + seed)
        sig = build_signals(book, ["ofi"])["ofi"]
        fwd = forward_returns(book, [10])["fwd_ret_10"]
        if block_permutation_pvalue(sig, fwd, spearman, block=500, n_perm=100, seed=1) < 0.05:
            rejects += 1
    assert rejects <= 2


def _gross_pnl(book):
    signals = build_signals(book, SIGNAL_SET)
    params = StrategyParams(signals=["ofi"], threshold=1.0, zscore_window=2000)
    free = CostModel(fixed_bps=0.0, half_spread_multiplier=0.0, impact_coefficient=0.0)
    return run_backtest(book, signals, params, free, ExecutionModel(level=1))["metrics"]["gross_pnl"]


def test_null_market_strategy_does_not_make_money_after_costs():
    book = simulate_book(n_events=30000, mode="null", seed=14)
    signals = build_signals(book, SIGNAL_SET)
    params = StrategyParams(signals=["ofi"], threshold=1.0, zscore_window=2000)
    metrics = run_backtest(book, signals, params, CostModel(), ExecutionModel(level=2))["metrics"]
    assert metrics["net_pnl"] < 0


def test_backtest_discriminates_between_a_planted_and_a_null_market():
    """The engine must earn more where an edge was planted than where none was."""
    planted = _gross_pnl(simulate_book(n_events=30000, mode="ofi", beta=0.45, seed=16))
    empty = _gross_pnl(simulate_book(n_events=30000, mode="null", seed=16))
    assert planted > empty
    assert planted > 0


def test_regime_market_concentrates_the_signal_in_wide_spreads():
    book = simulate_book(n_events=40000, mode="regime", beta=0.45, seed=15)
    signals = build_signals(book, ["ofi"])
    returns = forward_returns(book, [10])
    regimes = assign_regimes(book, window=2000)
    table = ic_by_regime(signals, returns, regimes, [10], min_obs=300)
    spread_cells = table[table["regime_dimension"] == "spread_regime"].set_index("regime")["rank_ic"]
    assert spread_cells["high"] > spread_cells["low"]
