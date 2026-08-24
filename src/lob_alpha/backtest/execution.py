"""Execution assumptions, as an explicit ladder.

Level 1  fill at mid, no cost. Not realistic, kept as the upper bound.
Level 2  cross the spread, pay the half spread.
Level 3  half spread plus slippage proportional to spread.
Level 4  partial fills: a fraction of the requested position change is filled.
Level 5  queue aware passive execution (stub, see the roadmap in the README).

The point of the ladder is that the paper can report the same strategy under
each assumption and show how much of the apparent edge is an artefact of the
assumption rather than the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..orderbook.representation import mid as _mid, spread as _spread

LEVELS = {
    1: "mid_fill",
    2: "cross_spread",
    3: "cross_spread_plus_slippage",
    4: "partial_fills",
    5: "queue_aware_passive",
}


@dataclass(frozen=True)
class ExecutionModel:
    level: int = 2
    slippage_fraction_of_spread: float = 0.25
    fill_ratio: float = 0.6  # level 4 only
    latency_events: int = 1  # a decision at t acts on the book at t + latency

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"execution level must be one of {sorted(LEVELS)}")
        if self.level == 5:
            raise NotImplementedError(
                "queue aware passive execution is a stretch goal, see README roadmap"
            )
        if self.latency_events < 1:
            raise ValueError("latency_events must be at least 1: a signal at t cannot trade at t")

    def realise(self, target_position: pd.Series, groups: pd.Series) -> pd.Series:
        """Turn a desired position into the position actually held.

        Latency is applied here, not in the signal, so that it is impossible to
        write a strategy that trades on information it did not yet have.
        """
        # Grouped by session: a position is never carried across a session
        # break by the latency shift, and every session starts flat.
        held = target_position.groupby(groups, sort=False).shift(self.latency_events).fillna(0.0)
        if self.level == 4:
            realised = np.zeros(len(held))
            values = held.to_numpy(float)
            group_values = groups.to_numpy()
            prev = 0.0
            prev_group = None
            for i in range(len(values)):
                if group_values[i] != prev_group:
                    prev, prev_group = 0.0, group_values[i]
                prev = prev + self.fill_ratio * (values[i] - prev)
                realised[i] = prev
            return pd.Series(realised, index=held.index)
        return held

    def cost_multiplier(self) -> float:
        """Extra cost on top of the cost model, from the execution assumption."""
        if self.level == 1:
            return 0.0
        if self.level == 3:
            return 1.0 + 2.0 * self.slippage_fraction_of_spread
        return 1.0

    def describe(self) -> str:
        return f"level {self.level} ({LEVELS[self.level]}), latency {self.latency_events} event(s)"


def effective_fill_price(book: pd.DataFrame, side: pd.Series, model: ExecutionModel) -> pd.Series:
    """Fill price for a trade of a given sign, used by the adverse selection study."""
    m = _mid(book)
    s = _spread(book)
    if model.level == 1:
        return m
    slip = model.slippage_fraction_of_spread * s if model.level == 3 else 0.0
    return m + np.sign(side) * (0.5 * s + slip)
