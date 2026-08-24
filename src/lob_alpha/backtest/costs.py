"""Transaction costs.

The model is deliberately parameterised rather than calibrated, because a
market impact coefficient estimated from public data is a guess wearing a lab
coat. The honest move is to state the functional form, sweep the coefficients,
and report where the strategy stops working.

Per unit of turnover, cost in return units:

    c = c0 + half_spread_multiplier * (s_t / 2m_t) + impact * (size / depth_t)

with ``c0`` in basis points covering commissions and fees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..orderbook.representation import depth as _depth, mid as _mid, spread as _spread


@dataclass(frozen=True)
class CostModel:
    fixed_bps: float = 0.0
    half_spread_multiplier: float = 1.0
    impact_coefficient: float = 0.0
    participation: float = 0.01  # order size as a fraction of displayed depth
    scale: float = 1.0  # global multiplier used by the cost sensitivity sweep

    def per_unit_turnover(self, book: pd.DataFrame) -> pd.Series:
        """Cost of trading one unit of position, in return units."""
        half_spread = 0.5 * _spread(book) / _mid(book)
        impact = self.impact_coefficient * self.participation * np.ones(len(book))
        cost = (
            self.fixed_bps * 1e-4
            + self.half_spread_multiplier * half_spread
            + pd.Series(impact, index=book.index)
        )
        return self.scale * cost

    def rescaled(self, scale: float) -> "CostModel":
        return CostModel(
            fixed_bps=self.fixed_bps,
            half_spread_multiplier=self.half_spread_multiplier,
            impact_coefficient=self.impact_coefficient,
            participation=self.participation,
            scale=scale,
        )
