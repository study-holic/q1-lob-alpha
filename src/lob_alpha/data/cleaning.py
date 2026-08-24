"""Deterministic cleaning.

Every rule removes rows for one stated reason and records how many. The audit
trail goes into the results directory, so the paper can state exactly what was
dropped rather than "the data were cleaned".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..orderbook.representation import L1_COLUMNS
from .sessions import group_key


def clean(
    raw: pd.DataFrame,
    drop_locked: bool = True,
    max_relative_jump: float = 0.05,
    session: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(clean_book, audit)``. The input frame is never mutated."""
    df = raw.copy()
    audit: list[dict] = []

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        mask = mask.fillna(True)
        n = int(mask.sum())
        audit.append({"rule": reason, "rows_dropped": n})
        df = df.loc[~mask]

    df = df.sort_values(["instrument", "timestamp"], kind="mergesort").reset_index(drop=True)

    drop(df[L1_COLUMNS].isna().any(axis=1), "null_l1_field")
    drop((df["bid_price"] <= 0) | (df["ask_price"] <= 0), "non_positive_price")
    drop((df["bid_size"] <= 0) | (df["ask_size"] <= 0), "non_positive_size")
    drop(df["ask_price"] < df["bid_price"], "crossed_book")
    if drop_locked:
        drop(df["ask_price"] == df["bid_price"], "locked_book")

    if "event_type" in df.columns:
        # LOBSTER type 7 marks a trading halt or resume. The book row attached
        # to it is a placeholder, not a quote.
        drop(df["event_type"] == 7, "trading_halt")

    drop(df.duplicated(subset=["instrument", "timestamp"], keep="last"), "duplicate_timestamp")

    if session is not None:
        tod = df["timestamp"].dt.time
        lo = pd.Timestamp(session[0]).time()
        hi = pd.Timestamp(session[1]).time()
        drop((tod < lo) | (tod > hi), "outside_session")

    mid = 0.5 * (df["bid_price"] + df["ask_price"])
    jump = mid.groupby(group_key(df), sort=False).pct_change().abs()
    drop(jump > max_relative_jump, "implausible_price_jump")

    # A row that is byte-identical to its predecessor carries no event.
    state = ["bid_price", "bid_size", "ask_price", "ask_size"]
    unchanged = (df[state] == df.groupby(group_key(df), sort=False)[state].shift()).all(axis=1)
    drop(unchanged.fillna(False), "stale_repeated_quote")

    df = df.sort_values(["instrument", "timestamp"], kind="mergesort").reset_index(drop=True)
    audit_df = pd.DataFrame(audit)
    audit_df["rows_dropped_pct"] = 100 * audit_df["rows_dropped"] / max(len(raw), 1)
    audit_df.attrs["rows_in"] = len(raw)
    audit_df.attrs["rows_out"] = len(df)
    return df, audit_df


def quality_report(book: pd.DataFrame) -> pd.DataFrame:
    """One row per instrument. This is the table that goes in the data section."""
    rows = []
    for name, g in book.groupby("instrument", sort=True):
        mid = 0.5 * (g["bid_price"] + g["ask_price"])
        spread = g["ask_price"] - g["bid_price"]
        gaps = g["timestamp"].diff().dt.total_seconds()
        rows.append(
            {
                "instrument": name,
                "observations": len(g),
                "start": g["timestamp"].min(),
                "end": g["timestamp"].max(),
                "missing_pct": 100 * g[L1_COLUMNS].isna().any(axis=1).mean(),
                "duplicate_ts_pct": 100 * g["timestamp"].duplicated().mean(),
                "median_spread": spread.median(),
                "p95_spread": spread.quantile(0.95),
                "median_rel_spread_bps": 1e4 * (spread / mid).median(),
                "median_depth": (g["bid_size"] + g["ask_size"]).median(),
                "price_min": mid.min(),
                "price_max": mid.max(),
                "median_gap_s": np.nanmedian(gaps),
                "max_gap_s": np.nanmax(gaps) if len(gaps.dropna()) else np.nan,
            }
        )
    return pd.DataFrame(rows)
