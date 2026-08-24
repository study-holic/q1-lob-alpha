"""Multiple testing and selection inflation.

Four signals by six horizons by nine regime cells by five instruments is 1,080
hypotheses before anyone has tried a second parameterisation. At the five
percent level that is 54 significant results from pure noise.

Two things happen here. First, corrections are applied to the IC table so the
paper can report how many results survive. Second, and more usefully, the
inflation is *measured*: the same selection procedure is run on data where the
answer is known to be nothing, and the resulting apparent performance is the
bar any real result has to clear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.significance import benjamini_hochberg


def correct_ic_table(ic_long: pd.DataFrame, alpha: float = 0.05, m: int | None = None) -> pd.DataFrame:
    """Attach BH and Bonferroni decisions to the IC table.

    ``m`` is the size of the hypothesis family the correction is taken over. It
    defaults to the number of rows, but that is almost never the honest number:
    the regime and per-instrument cells were tested too, and correcting only
    over the pooled table charges for a fraction of the searching actually
    done. Passing the full budget makes the correction conservative rather than
    flattering, which is the direction an error here should point.
    """
    if "p_value" not in ic_long or ic_long["p_value"].isna().all():
        return ic_long.assign(bh_reject=np.nan, bonferroni_reject=np.nan)
    m = int(m or len(ic_long))
    corrections = benjamini_hochberg(ic_long["p_value"], alpha=alpha, m=m)
    out = ic_long.join(corrections[["bh_threshold", "bh_reject", "bonferroni_reject"]])
    out["n_hypotheses"] = m
    out["n_tested_in_this_table"] = len(ic_long)
    return out


def hypothesis_count(n_signals: int, n_horizons: int, n_regime_cells: int, n_instruments: int) -> dict:
    total = n_signals * n_horizons * max(1, n_regime_cells) * max(1, n_instruments)
    return {
        "n_signals": n_signals,
        "n_horizons": n_horizons,
        "n_regime_cells": n_regime_cells,
        "n_instruments": n_instruments,
        "total_hypotheses": total,
        "expected_false_positives_at_5pct": 0.05 * total,
    }


def selection_inflation(
    scores_in_sample: pd.Series,
    scores_out_of_sample: pd.Series,
) -> dict:
    """Compare picking the winner in sample against its out-of-sample score.

    Both series are indexed by candidate specification. The gap between the
    best in-sample score and the out-of-sample score of that same candidate is
    the inflation attributable purely to selection.
    """
    df = pd.concat([scores_in_sample.rename("is"), scores_out_of_sample.rename("oos")], axis=1).dropna()
    if df.empty:
        return {}
    winner = df["is"].idxmax()
    return {
        "n_candidates": len(df),
        "winner": winner,
        "winner_in_sample": float(df.loc[winner, "is"]),
        "winner_out_of_sample": float(df.loc[winner, "oos"]),
        "inflation": float(df.loc[winner, "is"] - df.loc[winner, "oos"]),
        "mean_in_sample": float(df["is"].mean()),
        "mean_out_of_sample": float(df["oos"].mean()),
        "best_possible_out_of_sample": float(df["oos"].max()),
        "regret_vs_oracle": float(df["oos"].max() - df.loc[winner, "oos"]),
    }
