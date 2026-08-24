"""LOBSTER format loader.

LOBSTER ships two headerless CSVs per instrument-day, with one row in each per
book update:

* **message**: ``time, event_type, order_id, size, price, direction`` where
  ``time`` is seconds after midnight to nanosecond resolution.
* **orderbook**: ``ask_price_1, ask_size_1, bid_price_1, bid_size_1,
  ask_price_2, ...`` for as many levels as the sample provides.

Two traps worth naming. Prices are integers in units of 1/10000 of a dollar,
so a raw value of 2239500 is $223.95; forgetting the divisor silently rescales
every relative spread by four orders of magnitude. And LOBSTER pads an empty
book side with sentinel prices of -9999999999 or 9999999999, which pass a
naive positivity check on the ask side and would otherwise poison the mid.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_SCALE = 10_000.0
SENTINEL = 9_999_999_999
DEFAULT_DATE = "2012-06-21"


def load_lobster(
    orderbook_path: str | Path,
    message_path: str | Path,
    instrument: str,
    date: str = DEFAULT_DATE,
    levels: int = 1,
) -> pd.DataFrame:
    """Read one LOBSTER instrument-day into the canonical schema."""
    # Read only the levels requested. A 10-level file is 40 columns wide and a
    # full panel is well over half a gigabyte of CSV, most of it depth that a
    # level-1 experiment never touches.
    book = pd.read_csv(orderbook_path, header=None, usecols=range(4 * levels))
    messages = pd.read_csv(
        message_path,
        header=None,
        names=["time", "event_type", "order_id", "size", "price", "direction"],
        usecols=[0, 1, 2, 3, 4, 5],
    )
    if len(book) != len(messages):
        raise ValueError(
            f"row mismatch: {len(book)} book rows against {len(messages)} messages. "
            "The two files must come from the same LOBSTER export."
        )


    out = pd.DataFrame(
        {
            "timestamp": pd.Timestamp(date) + pd.to_timedelta(messages["time"].to_numpy(), unit="s"),
            "instrument": instrument,
            # Carried through so the cleaner can drop trading halts under a
            # named rule. LOBSTER writes type 7 on halt and resume, and the
            # corresponding book rows are not a real book state.
            "event_type": messages["event_type"].to_numpy(),
        }
    )
    for level in range(1, levels + 1):
        base = 4 * (level - 1)
        suffix = "" if level == 1 else f"_{level}"
        out[f"ask_price{suffix}"] = book.iloc[:, base] / PRICE_SCALE
        out[f"ask_size{suffix}"] = book.iloc[:, base + 1].astype(float)
        out[f"bid_price{suffix}"] = book.iloc[:, base + 2] / PRICE_SCALE
        out[f"bid_size{suffix}"] = book.iloc[:, base + 3].astype(float)

    # Sentinel-padded sides become missing, so the cleaner drops them under a
    # named rule rather than a mid-price of five hundred thousand dollars
    # quietly entering the return series.
    for col in [c for c in out.columns if "price" in c]:
        out.loc[out[col].abs() >= SENTINEL / PRICE_SCALE, col] = np.nan

    ordered = ["timestamp", "instrument", "bid_price", "bid_size", "ask_price", "ask_size", "event_type"]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]


def load_lobster_directory(path: str | Path, levels: int = 1) -> pd.DataFrame:
    """Load every message and orderbook pair under a directory into one panel.

    Files are paired by the LOBSTER naming convention, so a directory holding
    several tickers or several days concatenates into a multi-instrument,
    multi-session panel with no further configuration.
    """
    path = Path(path)
    books = sorted(path.rglob("*orderbook*.csv"))
    if not books:
        raise FileNotFoundError(f"no LOBSTER orderbook files under {path}")

    frames = []
    for book_path in books:
        message_path = Path(str(book_path).replace("orderbook", "message"))
        if not message_path.exists():
            raise FileNotFoundError(f"no message file paired with {book_path.name}")
        instrument, date = _parse_name(book_path.name)
        frames.append(load_lobster(book_path, message_path, instrument, date, levels))
    return pd.concat(frames, ignore_index=True)


def _parse_name(filename: str) -> tuple[str, str]:
    """``AMZN_2012-06-21_34200000_57600000_orderbook_5.csv`` to (AMZN, date).

    Falls back to the sample date rather than raising when a file has been
    renamed, since a wrong date shifts every timestamp by a constant and is
    harmless within a session, whereas refusing to load is not.
    """
    parts = filename.split("_")
    instrument = parts[0].upper() if parts else "UNKNOWN"
    date = next((p for p in parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p)), DEFAULT_DATE)
    return instrument, date
