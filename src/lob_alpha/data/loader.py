"""Loading raw books and mapping them onto the canonical schema.

Raw data is read-only. Cleaning writes to data/processed and never back.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..orderbook.representation import L1_COLUMNS
from . import synthetic
from .lobster import load_lobster_directory

DEFAULT_COLUMN_MAP = {
    "ts": "timestamp",
    "time": "timestamp",
    "symbol": "instrument",
    "ticker": "instrument",
    "bid": "bid_price",
    "ask": "ask_price",
    "bid_qty": "bid_size",
    "ask_qty": "ask_size",
    "bid_price_1": "bid_price",
    "ask_price_1": "ask_price",
    "bid_size_1": "bid_size",
    "ask_size_1": "ask_size",
}


def load_raw(config: dict) -> pd.DataFrame:
    """Load the dataset described by the ``data`` block of a config."""
    source = config.get("source", "synthetic")
    if source == "synthetic":
        syn = config.get("synthetic", {})
        return synthetic.simulate_panel(
            n_instruments=syn.get("n_instruments", 4),
            n_events=syn.get("n_events", 40_000),
            mode=syn.get("mode", "ofi"),
            beta=syn.get("beta", 0.12),
            seed=syn.get("seed", 0),
            n_sessions=syn.get("n_sessions", 1),
        )
    if source == "lobster":
        return load_lobster_directory(config["path"], levels=config.get("levels", 1))
    if source == "file":
        return load_files(config["path"], column_map=config.get("column_map"))
    raise ValueError(f"unknown data source: {source}")


def load_files(path: str | Path, column_map: dict | None = None) -> pd.DataFrame:
    """Read one file, or every csv/parquet under one directory."""
    path = Path(path)
    paths = sorted(p for p in path.rglob("*") if p.suffix in {".csv", ".parquet"}) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"no csv or parquet files under {path}")
    frames = [_read_one(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    return standardise(raw, column_map=column_map)


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def standardise(raw: pd.DataFrame, column_map: dict | None = None) -> pd.DataFrame:
    """Rename vendor columns onto the canonical schema and coerce dtypes."""
    mapping = dict(DEFAULT_COLUMN_MAP)
    mapping.update(column_map or {})
    df = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns}).copy()

    missing = [c for c in L1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"cannot map raw data onto the canonical schema, missing {missing}. "
            "Pass an explicit column_map in the config."
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["instrument"] = df["instrument"].astype(str)
    for col in ("bid_price", "bid_size", "ask_price", "ask_size"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    extras = [c for c in df.columns if c.startswith(("bid_price_", "ask_price_", "bid_size_", "ask_size_"))]
    return df[L1_COLUMNS + extras]
