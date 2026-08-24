"""Sessions: the unit that every shift, roll, and diff must respect.

An instrument is not a contiguous time series. It is a sequence of trading
sessions separated by gaps during which the price moves without a single
observed event. A 100-event forward return computed at 15:59 reaches across an
overnight gap and lands in the next morning, which is not a 100-event horizon;
it is an overnight return wearing a horizon's clothes. The same applies to
rolling z-scores, order flow accumulation, and position carry.

The fix is one grouping key, ``instrument|session``, used by every operation
that looks backwards or forwards in the frame. Nothing in the pipeline should
group by instrument alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_COLUMN = "session"
GROUP_COLUMN = "group_key"


def assign_sessions(
    book: pd.DataFrame,
    method: str = "calendar_day",
    max_gap_seconds: float = 1800.0,
) -> pd.Series:
    """Label each row with its trading session.

    ``calendar_day``  one session per instrument per date. Correct for most
                      equity data, where the exchange calendar does the work.
    ``gap``           a new session starts after a gap longer than
                      ``max_gap_seconds``. Correct for continuous venues such
                      as crypto, where there is no calendar boundary but there
                      are outages, and for any feed with unannounced downtime.
    """
    if method == "calendar_day":
        return book["timestamp"].dt.strftime("%Y-%m-%d")
    if method == "gap":
        gap = book.groupby("instrument", sort=False)["timestamp"].diff().dt.total_seconds()
        new_session = (gap > max_gap_seconds) | gap.isna()
        return new_session.groupby(book["instrument"], sort=False).cumsum().astype(int).astype(str)
    raise ValueError(f"unknown session method: {method!r}")


def with_sessions(book: pd.DataFrame, method: str = "calendar_day", **kwargs) -> pd.DataFrame:
    """Attach ``session`` and ``group_key`` columns. Idempotent."""
    out = book.copy()
    out[SESSION_COLUMN] = assign_sessions(out, method=method, **kwargs)
    out[GROUP_COLUMN] = out["instrument"].astype(str) + "|" + out[SESSION_COLUMN].astype(str)
    return out


def group_key(book: pd.DataFrame) -> pd.Series:
    """The grouping key every backward or forward looking operation must use.

    Falls back to instrument when sessions have not been assigned, so that
    existing single-session callers keep working, but the pipeline always
    assigns them.
    """
    if GROUP_COLUMN in book.columns:
        return book[GROUP_COLUMN]
    if SESSION_COLUMN in book.columns:
        return book["instrument"].astype(str) + "|" + book[SESSION_COLUMN].astype(str)
    return book["instrument"]


def session_report(book: pd.DataFrame) -> pd.DataFrame:
    """One row per session. Short sessions distort event-time horizons."""
    keys = group_key(book)
    rows = []
    for key, g in book.groupby(keys, sort=True):
        gaps = g["timestamp"].diff().dt.total_seconds()
        rows.append(
            {
                "group_key": key,
                "instrument": g["instrument"].iloc[0],
                "observations": len(g),
                "start": g["timestamp"].min(),
                "end": g["timestamp"].max(),
                "duration_minutes": (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 60,
                "median_gap_s": float(np.nanmedian(gaps)) if len(gaps.dropna()) else np.nan,
                "max_gap_s": float(np.nanmax(gaps)) if len(gaps.dropna()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def drop_short_sessions(book: pd.DataFrame, min_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove sessions too short to support the longest horizon under test.

    A session with fewer events than the maximum horizon contributes only
    missing forward returns, so it adds rows without adding evidence.
    """
    keys = group_key(book)
    counts = keys.map(keys.value_counts())
    keep = counts >= min_events
    dropped = pd.DataFrame(
        {
            "rule": ["session_shorter_than_max_horizon"],
            "rows_dropped": [int((~keep).sum())],
            "sessions_dropped": [int(keys[~keep].nunique())],
        }
    )
    return book.loc[keep].reset_index(drop=True), dropped
