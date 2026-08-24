"""Spread.

    s_t = P^a_t - P^b_t   and   s_t / m_t

Spread is included as a candidate directional predictor, but the prior is that
it earns its place as a *conditioning* variable. The regime analysis tests that
directly, and the redundancy analysis asks whether spread adds anything once
the other three signals are known.
"""

from __future__ import annotations

import pandas as pd

from ..orderbook.representation import mid as _mid, relative_spread, spread as _spread


def spread_signal(book: pd.DataFrame, relative: bool = True, demean_window: int = 0) -> pd.Series:
    s = relative_spread(book) if relative else _spread(book)
    if demean_window > 0:
        s = s - s.rolling(demean_window, min_periods=demean_window // 5).mean()
    return s.rename("spread")
