"""Micro-price deviation.

    MicroPrice_t = (P^a_t Q^b_t + P^b_t Q^a_t) / (Q^b_t + Q^a_t)
    MPD_t        = MicroPrice_t - m_t

The deviation, not the level, is the signal: the level is dominated by the
price itself and is non-stationary. Reporting it in relative terms keeps it
comparable across instruments at different price points.
"""

from __future__ import annotations

import pandas as pd

from ..orderbook.representation import microprice as _microprice, mid as _mid


def microprice_deviation(book: pd.DataFrame, relative: bool = True) -> pd.Series:
    m = _mid(book)
    dev = _microprice(book) - m
    if relative:
        dev = dev / m
    return dev.rename("microprice_deviation")
