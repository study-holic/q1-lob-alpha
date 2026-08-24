"""Queue imbalance.

    QI_t = (Q^b_t - Q^a_t) / (Q^b_t + Q^a_t),  QI in [-1, 1]

+1 means all displayed top-of-book liquidity sits on the bid, -1 all on the
ask, 0 balanced. Whether positive QI precedes a rise is an empirical question
this project answers rather than assumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def queue_imbalance(book: pd.DataFrame, levels: int = 1) -> pd.Series:
    qb = book["bid_size"].astype(float).copy()
    qa = book["ask_size"].astype(float).copy()
    for i in range(2, levels + 1):
        if f"bid_size_{i}" in book:
            qb = qb + book[f"bid_size_{i}"].fillna(0.0)
        if f"ask_size_{i}" in book:
            qa = qa + book[f"ask_size_{i}"].fillna(0.0)
    denom = (qb + qa).replace(0.0, np.nan)
    return ((qb - qa) / denom).rename("queue_imbalance")
