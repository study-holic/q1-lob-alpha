"""Signal mathematics, including hand-constructed OFI cases."""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.signals import build_signals
from lob_alpha.signals.microprice import microprice_deviation
from lob_alpha.signals.ofi import ofi, ofi_increments
from lob_alpha.signals.queue_imbalance import queue_imbalance


def book_from(rows):
    ts = pd.date_range("2026-01-01 09:00:00", periods=len(rows), freq="100ms")
    df = pd.DataFrame(rows, columns=["bid_price", "bid_size", "ask_price", "ask_size"])
    df.insert(0, "instrument", "X")
    df.insert(0, "timestamp", ts)
    return df


def test_queue_imbalance_bounds_and_signs(small_book):
    qi = queue_imbalance(small_book).dropna()
    assert qi.between(-1, 1).all()
    book = book_from([[99.99, 900.0, 100.01, 100.0], [99.99, 100.0, 100.01, 900.0]])
    qi = queue_imbalance(book)
    assert qi.iloc[0] == pytest.approx(0.8)
    assert qi.iloc[1] == pytest.approx(-0.8)


def test_microprice_deviation_sign_follows_imbalance():
    book = book_from([[99.99, 900.0, 100.01, 100.0], [99.99, 100.0, 100.01, 900.0]])
    dev = microprice_deviation(book)
    assert dev.iloc[0] > 0
    assert dev.iloc[1] < 0


# --- OFI: each branch of the definition, by hand ---------------------------

def test_ofi_bid_size_increase_is_positive():
    book = book_from([[99.99, 100.0, 100.01, 100.0], [99.99, 180.0, 100.01, 100.0]])
    assert ofi_increments(book).iloc[1] == pytest.approx(80.0)


def test_ofi_ask_size_increase_is_negative():
    book = book_from([[99.99, 100.0, 100.01, 100.0], [99.99, 100.0, 100.01, 180.0]])
    assert ofi_increments(book).iloc[1] == pytest.approx(-80.0)


def test_ofi_bid_price_increase_counts_the_whole_new_queue():
    book = book_from([[99.99, 100.0, 100.01, 100.0], [100.00, 70.0, 100.02, 100.0]])
    # bid up: +70. ask up: the old ask queue was lifted, so -(-100) = +100.
    assert ofi_increments(book).iloc[1] == pytest.approx(170.0)


def test_ofi_ask_price_decrease_is_negative():
    book = book_from([[99.99, 100.0, 100.01, 100.0], [99.98, 100.0, 100.00, 60.0]])
    # bid down: -100. ask down: new ask queue is fresh supply, -60.
    assert ofi_increments(book).iloc[1] == pytest.approx(-160.0)


def test_ofi_is_antisymmetric_under_side_reflection():
    book = book_from([[99.99, 100.0, 100.01, 150.0], [99.99, 260.0, 100.01, 90.0]])
    mirrored = book.copy()
    mirrored[["bid_size", "ask_size"]] = book[["ask_size", "bid_size"]].to_numpy()
    assert ofi_increments(book).iloc[1] == pytest.approx(-ofi_increments(mirrored).iloc[1])


def test_ofi_first_event_of_each_instrument_is_zero(small_panel):
    inc = ofi_increments(small_panel)
    firsts = small_panel.groupby("instrument", sort=False).head(1).index
    assert (inc.loc[firsts] == 0).all()


def test_ofi_does_not_leak_across_instruments(small_panel):
    joined = ofi(small_panel, window=20)
    separate = pd.concat(
        [ofi(g.reset_index(drop=True), window=20) for _, g in small_panel.groupby("instrument", sort=False)],
        ignore_index=True,
    )
    np.testing.assert_allclose(joined.to_numpy(), separate.to_numpy(), equal_nan=True, rtol=1e-9)


def test_registry_builds_all_four_signals(small_book):
    sigs = build_signals(small_book, ["ofi", "queue_imbalance", "microprice_deviation", "spread"])
    assert list(sigs.columns) == ["ofi", "queue_imbalance", "microprice_deviation", "spread"]
    assert sigs.notna().sum().min() > 0


def test_unknown_signal_raises(small_book):
    with pytest.raises(KeyError):
        build_signals(small_book, ["not_a_signal"])
