"""Forward returns must look forward, and only forward."""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.orderbook.representation import mid
from lob_alpha.targets.forward_returns import forward_returns, horizon_clock_time


def test_forward_return_matches_manual_calculation(small_book):
    fwd = forward_returns(small_book, [5])["fwd_ret_5"]
    m = mid(small_book)
    expected = (m.shift(-5) - m) / m
    pd.testing.assert_series_equal(fwd, expected, check_names=False)


def test_forward_return_tail_is_missing_not_zero(small_book):
    fwd = forward_returns(small_book, [10])["fwd_ret_10"]
    assert fwd.tail(10).isna().all()


def test_no_leakage_across_instruments(small_panel):
    fwd = forward_returns(small_panel, [3])["fwd_ret_3"]
    last_rows = small_panel.groupby("instrument", sort=False).tail(3).index
    assert fwd.loc[last_rows].isna().all()


def test_horizon_clock_time_is_monotone(small_book):
    clock = horizon_clock_time(small_book, [1, 5, 10, 25])
    assert clock["median_seconds"].is_monotonic_increasing


def test_shuffling_the_future_destroys_the_relationship(small_book):
    """A sanity check on the check: the target must actually carry signal."""
    from lob_alpha.evaluation.ic import rank_ic
    from lob_alpha.signals.ofi import ofi

    sig = ofi(small_book, window=20)
    fwd = forward_returns(small_book, [10])["fwd_ret_10"]
    real = abs(rank_ic(sig, fwd))
    scrambled = abs(rank_ic(sig, pd.Series(np.roll(fwd.to_numpy(), 1234), index=fwd.index)))
    assert real > scrambled
