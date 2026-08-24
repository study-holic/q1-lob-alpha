"""Session boundaries: nothing may look across an overnight gap.

These are the tests that would have caught the bug. On single-session data
every one of them passes trivially, which is exactly why the synthetic
generator now produces multi-session panels by default in these fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.backtest.costs import CostModel
from lob_alpha.backtest.engine import StrategyParams, build_position, run_backtest
from lob_alpha.backtest.execution import ExecutionModel
from lob_alpha.data.sessions import (
    assign_sessions,
    drop_short_sessions,
    group_key,
    session_report,
    with_sessions,
)
from lob_alpha.data.synthetic import simulate_book, simulate_panel
from lob_alpha.evaluation.regimes import assign_regimes
from lob_alpha.orderbook.representation import mid
from lob_alpha.signals import build_signals
from lob_alpha.signals.ofi import ofi_increments
from lob_alpha.targets.forward_returns import forward_returns

HORIZONS = [1, 5, 10]


@pytest.fixture(scope="module")
def multi_session():
    book = simulate_panel(n_instruments=2, n_events=6000, n_sessions=3, seed=4)
    return with_sessions(book)


def test_sessions_split_on_calendar_day(multi_session):
    assert multi_session["session"].nunique() == 3
    assert group_key(multi_session).nunique() == 6


def test_gap_method_splits_on_downtime():
    book = simulate_book(n_events=400, seed=1)
    book.loc[200:, "timestamp"] += pd.Timedelta(hours=4)
    labels = assign_sessions(book, method="gap", max_gap_seconds=1800)
    assert labels.nunique() == 2
    assert labels.iloc[199] != labels.iloc[200]


def test_forward_returns_never_span_a_session(multi_session):
    """The core bug: a 10-event horizon at the close is an overnight return."""
    fwd = forward_returns(multi_session, HORIZONS)
    keys = group_key(multi_session)
    for h in HORIZONS:
        tail = multi_session.groupby(keys, sort=False).tail(h).index
        assert fwd.loc[tail, f"fwd_ret_{h}"].isna().all()


def test_forward_return_values_match_within_session(multi_session):
    fwd = forward_returns(multi_session, [5])["fwd_ret_5"]
    m = mid(multi_session)
    expected = m.groupby(group_key(multi_session), sort=False).shift(-5)
    expected = (expected - m) / m
    pd.testing.assert_series_equal(fwd, expected, check_names=False)


def test_ofi_resets_at_every_session_start(multi_session):
    inc = ofi_increments(multi_session)
    firsts = multi_session.groupby(group_key(multi_session), sort=False).head(1).index
    assert (inc.loc[firsts] == 0).all()


def test_overnight_gap_would_be_visible_without_the_fix(multi_session):
    """Sanity check on the fixture: the boundary is a real discontinuity.

    If the overnight gap were negligible, these tests would pass on broken
    code, so the generator's gap has to be large relative to an event move.
    """
    m = mid(multi_session)
    keys = group_key(multi_session)
    naive = m.groupby(multi_session["instrument"], sort=False).pct_change().abs()
    boundary = keys != keys.shift()
    within = naive[~boundary & multi_session["instrument"].eq(multi_session["instrument"].shift())]
    crossing = naive[boundary & multi_session["instrument"].eq(multi_session["instrument"].shift())]
    assert crossing.max() > 10 * within.median()


def test_position_starts_flat_in_every_session(multi_session):
    signals = build_signals(multi_session, ["ofi"])
    params = StrategyParams(signals=["ofi"], threshold=0.5, zscore_window=200)
    keys = group_key(multi_session)
    target = build_position(signals, keys, params)
    held = ExecutionModel(level=2, latency_events=1).realise(target, keys)
    firsts = multi_session.groupby(keys, sort=False).head(1).index
    assert (held.loc[firsts] == 0).all()


def test_no_pnl_is_earned_across_a_session_boundary(multi_session):
    signals = build_signals(multi_session, ["ofi"])
    params = StrategyParams(signals=["ofi"], threshold=0.5, zscore_window=200)
    result = run_backtest(multi_session, signals, params, CostModel(), ExecutionModel(level=2))
    keys = group_key(multi_session)
    lasts = multi_session.groupby(keys, sort=False).tail(1).index
    assert (result["pnl"].loc[lasts, "gross"] == 0).all()


def test_regime_labels_do_not_carry_across_sessions(multi_session):
    """A session's first events have no trailing window of their own yet."""
    regimes = assign_regimes(multi_session, window=300, vol_window=50)
    keys = group_key(multi_session)
    firsts = multi_session.groupby(keys, sort=False).head(30).index
    assert regimes.loc[firsts, "spread_regime"].isna().all()


def test_short_sessions_are_dropped_with_an_audit_trail(multi_session):
    truncated = pd.concat(
        [multi_session, multi_session.head(5).assign(session="1999-01-01", group_key="SYN1|1999-01-01")],
        ignore_index=True,
    )
    kept, audit = drop_short_sessions(truncated, min_events=100)
    assert "SYN1|1999-01-01" not in set(group_key(kept))
    assert audit["rows_dropped"].iloc[0] == 5
    assert audit["sessions_dropped"].iloc[0] == 1


def test_session_report_covers_every_session(multi_session):
    report = session_report(multi_session)
    assert len(report) == group_key(multi_session).nunique()
    assert (report["observations"] > 0).all()
