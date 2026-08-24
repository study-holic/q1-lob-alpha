"""Canonical limit order book representation.

The research pipeline never touches raw vendor columns directly. Everything
flows through the canonical schema defined here, and every downstream module is
entitled to assume the invariants in :func:`validate_invariants` hold.

Canonical columns (level 1 is mandatory, deeper levels optional):

    timestamp, instrument, bid_price, bid_size, ask_price, ask_size
    bid_price_2 ... bid_price_N, bid_size_2 ... (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

L1_COLUMNS = [
    "timestamp",
    "instrument",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
]


@dataclass(frozen=True)
class OrderBookSnapshot:
    """A single point-in-time view of the book. Used in tests and examples."""

    timestamp: pd.Timestamp
    instrument: str
    bids: Sequence[tuple[float, float]]  # [(price, size), ...] best first
    asks: Sequence[tuple[float, float]]

    @property
    def best_bid(self) -> tuple[float, float]:
        return self.bids[0]

    @property
    def best_ask(self) -> tuple[float, float]:
        return self.asks[0]

    @property
    def mid(self) -> float:
        return 0.5 * (self.bids[0][0] + self.asks[0][0])

    @property
    def spread(self) -> float:
        return self.asks[0][0] - self.bids[0][0]

    def to_row(self) -> dict:
        row = {
            "timestamp": self.timestamp,
            "instrument": self.instrument,
            "bid_price": self.bids[0][0],
            "bid_size": self.bids[0][1],
            "ask_price": self.asks[0][0],
            "ask_size": self.asks[0][1],
        }
        for i, (p, q) in enumerate(self.bids[1:], start=2):
            row[f"bid_price_{i}"] = p
            row[f"bid_size_{i}"] = q
        for i, (p, q) in enumerate(self.asks[1:], start=2):
            row[f"ask_price_{i}"] = p
            row[f"ask_size_{i}"] = q
        return row


def snapshots_to_frame(snapshots: Sequence[OrderBookSnapshot]) -> pd.DataFrame:
    return pd.DataFrame([s.to_row() for s in snapshots])


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------

def mid(book: pd.DataFrame) -> pd.Series:
    """m_t = (P^b_t + P^a_t) / 2."""
    return 0.5 * (book["bid_price"] + book["ask_price"])


def spread(book: pd.DataFrame) -> pd.Series:
    """s_t = P^a_t - P^b_t."""
    return book["ask_price"] - book["bid_price"]


def relative_spread(book: pd.DataFrame) -> pd.Series:
    """s_t / m_t."""
    return spread(book) / mid(book)


def depth(book: pd.DataFrame, levels: int = 1) -> pd.Series:
    """D_t = sum of displayed size on both sides over `levels` levels."""
    total = book["bid_size"] + book["ask_size"]
    for i in range(2, levels + 1):
        bcol, acol = f"bid_size_{i}", f"ask_size_{i}"
        if bcol in book:
            total = total + book[bcol].fillna(0.0)
        if acol in book:
            total = total + book[acol].fillna(0.0)
    return total


def microprice(book: pd.DataFrame) -> pd.Series:
    """Size weighted mid: (P^a Q^b + P^b Q^a) / (Q^b + Q^a).

    Weighting the ask price by *bid* size is deliberate: a large bid queue pulls
    the fair price towards the ask.
    """
    qb, qa = book["bid_size"], book["ask_size"]
    denom = (qb + qa).replace(0.0, np.nan)
    return (book["ask_price"] * qb + book["bid_price"] * qa) / denom


def realised_volatility(book: pd.DataFrame, window: int = 200, groups: pd.Series | None = None) -> pd.Series:
    """Rolling standard deviation of mid log returns, in event time.

    The first return of each session is dropped rather than measured against
    the previous close, since an overnight move is not an event-time return.
    """
    m = mid(book)
    if groups is None:
        groups = book["instrument"]
    r = np.log(m).groupby(groups, sort=False).diff()
    return r.groupby(groups, sort=False).transform(
        lambda s: s.rolling(window, min_periods=max(10, window // 10)).std()
    )


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------

def invariant_violations(book: pd.DataFrame) -> pd.DataFrame:
    """Return a per-invariant count of violations. Empty frame means clean."""
    checks = {
        "crossed_book": book["ask_price"] < book["bid_price"],
        "locked_book": book["ask_price"] == book["bid_price"],
        "non_positive_bid_size": book["bid_size"] <= 0,
        "non_positive_ask_size": book["ask_size"] <= 0,
        "non_positive_price": (book["bid_price"] <= 0) | (book["ask_price"] <= 0),
        "null_fields": book[L1_COLUMNS].isna().any(axis=1),
    }
    rows = [
        {"invariant": name, "violations": int(flag.sum())}
        for name, flag in checks.items()
    ]
    out = pd.DataFrame(rows)
    return out[out["violations"] > 0].reset_index(drop=True)


def validate_invariants(book: pd.DataFrame) -> None:
    """Raise if the canonical schema or its invariants are broken."""
    missing = [c for c in L1_COLUMNS if c not in book.columns]
    if missing:
        raise ValueError(f"book is missing canonical columns: {missing}")
    bad = invariant_violations(book)
    if not bad.empty:
        raise ValueError(f"order book invariants violated:\n{bad.to_string(index=False)}")
    # The panel is stored instrument-major, so monotonicity is a within
    # instrument property. Checking it globally would either fail on a valid
    # panel or force a sort order that hides genuinely out of order events.
    from ..data.sessions import group_key

    ordered = book.groupby(group_key(book), sort=False)["timestamp"].apply(
        lambda s: s.is_monotonic_increasing
    )
    if not ordered.all():
        bad = list(ordered[~ordered].index)
        raise ValueError(f"timestamps are not monotonically increasing for: {bad}")
