"""Order flow imbalance.

The naive definition, the change in bid size minus the change in ask size,
conflates three very different events: a queue growing because orders arrived,
a queue shrinking because orders were cancelled, and a queue vanishing because
the price level moved. The Cont, Kukanov and Stoikov construction handles this
by conditioning on the price move.

For consecutive book states n-1 and n, the bid side contribution is

    e^b_n = 1{P^b_n >= P^b_{n-1}} Q^b_n  -  1{P^b_n <= P^b_{n-1}} Q^b_{n-1}

which reads as three cases:

    bid price up    : the whole new queue is new demand      -> + Q^b_n
    bid price flat  : only the change in queue size matters  -> + (Q^b_n - Q^b_{n-1})
    bid price down  : the old queue was consumed or pulled   -> - Q^b_{n-1}

The ask side mirrors it with the inequalities reversed, and

    OFI_n = e^b_n - e^a_n

so buying pressure is positive by construction. The raw increment is noisy at
event frequency, so the reported signal is a rolling sum over `window` events,
normalised by trailing mean depth over `depth_window` events to keep it comparable across instruments
and across time within an instrument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.sessions import group_key
from ..orderbook.representation import depth


def ofi_increments(book: pd.DataFrame) -> pd.Series:
    """Per-event OFI increment e_n. First event of each instrument is 0."""
    # Grouped by instrument *and session*: an OFI increment must never be
    # computed across an overnight gap, where the previous book state is
    # hours stale and the price has moved without a single observed event.
    g = book.groupby(group_key(book), sort=False)
    pb0, qb0 = g["bid_price"].shift(), g["bid_size"].shift()
    pa0, qa0 = g["ask_price"].shift(), g["ask_size"].shift()
    pb1, qb1 = book["bid_price"], book["bid_size"]
    pa1, qa1 = book["ask_price"], book["ask_size"]

    e_bid = np.where(pb1 >= pb0, qb1, 0.0) - np.where(pb1 <= pb0, qb0, 0.0)
    e_ask = np.where(pa1 <= pa0, qa1, 0.0) - np.where(pa1 >= pa0, qa0, 0.0)

    out = pd.Series(e_bid - e_ask, index=book.index, name="ofi_increment")
    return out.fillna(0.0)


def ofi(
    book: pd.DataFrame,
    window: int = 20,
    normalise: bool = True,
    depth_window: int = 200,
) -> pd.Series:
    """Rolling order flow imbalance over `window` events."""
    inc = ofi_increments(book)
    rolled = inc.groupby(group_key(book), sort=False).transform(
        lambda s: s.rolling(window, min_periods=max(2, window // 4)).sum()
    )
    if normalise:
        d = depth(book).groupby(group_key(book), sort=False).transform(
            lambda s: s.rolling(depth_window, min_periods=max(10, depth_window // 10)).mean()
        )
        rolled = rolled / d.replace(0.0, np.nan)
    return rolled.rename("ofi")
