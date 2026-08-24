"""Invariants of the canonical book representation."""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.orderbook.representation import (
    OrderBookSnapshot,
    depth,
    invariant_violations,
    microprice,
    mid,
    relative_spread,
    snapshots_to_frame,
    spread,
    validate_invariants,
)


def test_synthetic_book_satisfies_invariants(small_book):
    validate_invariants(small_book)
    assert invariant_violations(small_book).empty


def test_spread_is_non_negative_and_mid_is_between_quotes(small_book):
    s, m = spread(small_book), mid(small_book)
    assert (s >= 0).all()
    assert (m >= small_book["bid_price"]).all()
    assert (m <= small_book["ask_price"]).all()


def test_microprice_lies_between_bid_and_ask(small_book):
    mp = microprice(small_book)
    assert (mp >= small_book["bid_price"] - 1e-12).all()
    assert (mp <= small_book["ask_price"] + 1e-12).all()


def test_zero_imbalance_gives_microprice_equal_to_mid():
    book = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:00:00"]),
            "instrument": ["X"],
            "bid_price": [99.99],
            "bid_size": [100.0],
            "ask_price": [100.01],
            "ask_size": [100.0],
        }
    )
    assert microprice(book).iloc[0] == pytest.approx(mid(book).iloc[0])


def test_microprice_leans_towards_the_larger_queue():
    book = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:00:00"]),
            "instrument": ["X"],
            "bid_price": [99.99],
            "bid_size": [900.0],
            "ask_price": [100.01],
            "ask_size": [100.0],
        }
    )
    assert microprice(book).iloc[0] > mid(book).iloc[0]


def test_relative_spread_and_depth():
    book = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:00:00"]),
            "instrument": ["X"],
            "bid_price": [99.0],
            "bid_size": [10.0],
            "ask_price": [101.0],
            "ask_size": [30.0],
        }
    )
    assert relative_spread(book).iloc[0] == pytest.approx(2.0 / 100.0)
    assert depth(book).iloc[0] == 40.0


def test_crossed_book_is_rejected(small_book):
    bad = small_book.copy()
    bad.loc[bad.index[10], "ask_price"] = bad.loc[bad.index[10], "bid_price"] - 0.05
    with pytest.raises(ValueError, match="invariants"):
        validate_invariants(bad)


def test_snapshot_roundtrip():
    snap = OrderBookSnapshot(
        timestamp=pd.Timestamp("2026-01-01 09:00:00"),
        instrument="X",
        bids=[(99.99, 100.0), (99.98, 250.0)],
        asks=[(100.01, 120.0), (100.02, 300.0)],
    )
    frame = snapshots_to_frame([snap])
    assert frame["bid_price_2"].iloc[0] == 99.98
    assert snap.mid == pytest.approx(100.0)
    assert snap.spread == pytest.approx(0.02)
