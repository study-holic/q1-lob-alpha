"""Signal registry.

Adding a signal here is the only way to make it available to a config. That is
deliberate: the protocol freezes the signal set at four, and expanding it
should be a visible decision rather than a notebook side effect.
"""

from __future__ import annotations

import pandas as pd

from .microprice import microprice_deviation
from .ofi import ofi, ofi_increments
from .queue_imbalance import queue_imbalance
from .spread import spread_signal

REGISTRY = {
    "ofi": ofi,
    "queue_imbalance": queue_imbalance,
    "microprice_deviation": microprice_deviation,
    "spread": spread_signal,
}

__all__ = [
    "REGISTRY",
    "build_signals",
    "microprice_deviation",
    "ofi",
    "ofi_increments",
    "queue_imbalance",
    "spread_signal",
]


def build_signals(book: pd.DataFrame, spec) -> pd.DataFrame:
    """Compute the requested signals.

    ``spec`` is either a list of names or a mapping ``name -> kwargs``.
    """
    if isinstance(spec, list):
        spec = {name: {} for name in spec}
    out = {}
    for name, kwargs in spec.items():
        if name not in REGISTRY:
            raise KeyError(f"unknown signal {name!r}; registered: {sorted(REGISTRY)}")
        out[name] = REGISTRY[name](book, **(kwargs or {}))
    return pd.DataFrame(out, index=book.index)
