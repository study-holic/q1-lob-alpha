"""Forward returns.

    r_{t,h} = (m_{t+h} - m_t) / m_t

The horizon h is measured in events. Because event frequency varies with the
regime, every horizon is also reported as elapsed clock time, so a statement
like "the signal dies after 25 events" can be translated into seconds, and so
that two regimes with different event rates are not silently compared.

Horizons are shifted *within* instrument. Shifting across a concatenated panel
is one of the easiest ways to manufacture a look-ahead bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.sessions import group_key
from ..orderbook.representation import mid as _mid


def forward_returns(book: pd.DataFrame, horizons: list[int], use_log: bool = False) -> pd.DataFrame:
    m = _mid(book)
    # Session-aware: a horizon that would reach past the close returns NaN
    # rather than an overnight return mislabelled as an h-event move.
    g = m.groupby(group_key(book), sort=False)
    out = {}
    for h in horizons:
        fwd = g.shift(-h)
        out[f"fwd_ret_{h}"] = np.log(fwd / m) if use_log else (fwd - m) / m
    return pd.DataFrame(out, index=book.index)


def horizon_clock_time(book: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Median and mean elapsed seconds corresponding to each event horizon."""
    ts = book["timestamp"]
    g = ts.groupby(group_key(book), sort=False)
    rows = []
    for h in horizons:
        elapsed = (g.shift(-h) - ts).dt.total_seconds()
        rows.append(
            {
                "horizon_events": h,
                "median_seconds": float(np.nanmedian(elapsed)),
                "mean_seconds": float(np.nanmean(elapsed)),
                "p95_seconds": float(np.nanpercentile(elapsed.dropna(), 95)) if elapsed.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
