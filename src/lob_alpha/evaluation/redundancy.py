"""Are four signals four pieces of information?

Three questions, in order:

1. How correlated are the signals with each other, in level and in rank?
2. How much of each signal is explained by the others (VIF)?
3. Does adding a signal to a nested model raise out-of-sample R squared?

Question 3 is the one that matters. In-sample R squared cannot fall when a
regressor is added, so an in-sample nested comparison always rewards
complexity. The incremental R squared reported here is measured on a held-out
tail of the sample, which can and often does go negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrices(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = signals.dropna()
    return clean.corr(method="pearson"), clean.corr(method="spearman")


def variance_inflation(signals: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R2_j) where R2_j regresses signal j on the others."""
    clean = signals.dropna()
    rows = []
    for col in clean.columns:
        others = clean.drop(columns=[col])
        r2 = _ols_r2(others.to_numpy(float), clean[col].to_numpy(float))
        rows.append({"signal": col, "r2_on_others": r2, "vif": np.inf if r2 >= 1 else 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows)


def _design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(_design(x), y, rcond=None)[0]


def _ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    beta = _ols_fit(x, y)
    resid = y - _design(x) @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - float((resid**2).sum()) / ss_tot


def _oos_r2(x_tr, y_tr, x_te, y_te) -> float:
    beta = _ols_fit(x_tr, y_tr)
    pred = _design(x_te) @ beta
    ss_tot = float(((y_te - y_tr.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - float(((y_te - pred) ** 2).sum()) / ss_tot


def nested_models(
    signals: pd.DataFrame,
    forward_return: pd.Series,
    order: list[str] | None = None,
    test_fraction: float = 0.3,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Add signals one at a time and report in-sample and held-out R squared.

    The split is the chronological tail **within each instrument**. Taking the
    tail of a concatenated panel instead would hand the entire test set to
    whichever instrument sits last in the frame, so the reported increment
    would measure how the signals rank on that one name rather than how they
    generalise.

    Increments are order-dependent whenever two regressors are collinear: the
    one entering first absorbs the shared explanatory power. Pass ``order``
    deliberately, and see :func:`nested_models_both_orders`.
    """
    order = order or list(signals.columns)
    frame = {c: signals[c] for c in order}
    frame["y"] = forward_return
    if groups is not None:
        frame["__g"] = groups
    df = pd.DataFrame(frame).dropna()
    if len(df) < 200:
        return pd.DataFrame()

    if groups is None:
        is_test = np.zeros(len(df), dtype=bool)
        is_test[int(len(df) * (1 - test_fraction)):] = True
    else:
        is_test = np.zeros(len(df), dtype=bool)
        pos = np.arange(len(df))
        for _, idx in df.groupby("__g", sort=False).indices.items():
            k = int(len(idx) * (1 - test_fraction))
            is_test[idx[k:]] = True
        df = df.drop(columns="__g")

    y = df["y"].to_numpy(float)
    y_tr, y_te = y[~is_test], y[is_test]
    cut = int((~is_test).sum())

    rows = []
    prev_oos = 0.0
    for k in range(1, len(order) + 1):
        cols = order[:k]
        x = df[cols].to_numpy(float)
        r2_is = _ols_r2(x[~is_test], y_tr)
        r2_oos = _oos_r2(x[~is_test], y_tr, x[is_test], y_te)
        rows.append(
            {
                "model": f"M{k}",
                "signals": " + ".join(cols),
                "n_train": int((~is_test).sum()),
                "n_test": int(is_test.sum()),
                "r2_in_sample": r2_is,
                "r2_out_of_sample": r2_oos,
                "incremental_oos_r2": r2_oos - prev_oos,
            }
        )
        prev_oos = r2_oos
    return pd.DataFrame(rows)


def quantile_response(
    signal: pd.Series,
    forward_return: pd.Series,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Mean forward return by signal quantile, with a standard error.

    This is where a non-monotone or threshold relationship shows up, and it is
    the usual explanation for a weak looking rank IC sitting on top of a real
    effect.
    """
    df = pd.concat([signal.rename("s"), forward_return.rename("r")], axis=1).dropna()
    if len(df) < 100 or df["s"].nunique() < n_buckets:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["s"].rank(method="first"), n_buckets, labels=False) + 1
    g = df.groupby("bucket")["r"]
    out = g.agg(n="size", mean_forward_return="mean", std="std").reset_index()
    out["stderr"] = out["std"] / np.sqrt(out["n"])
    out["signal_mean"] = df.groupby("bucket")["s"].mean().to_numpy()
    monotone = out["mean_forward_return"].is_monotonic_increasing or out["mean_forward_return"].is_monotonic_decreasing
    out.attrs["monotone"] = bool(monotone)
    return out


def nested_models_both_orders(
    signals: pd.DataFrame,
    forward_return: pd.Series,
    collinear_pair: tuple[str, str],
    base: str,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the nested sequence with a collinear pair entered in both orders.

    When two regressors are nearly collinear, the incremental R squared of the
    second is not a property of that signal; it is a property of the ordering.
    Reporting both orders is the cheapest way to make that visible instead of
    writing "X adds nothing" when the truth is "X adds nothing once Y is in".
    """
    a, b = collinear_pair
    rest = [c for c in signals.columns if c not in (a, b, base)]
    out = []
    for first, second in ((a, b), (b, a)):
        table = nested_models(signals, forward_return, order=[base, first, second, *rest], groups=groups)
        if len(table):
            table.insert(0, "entry_order", f"{first} before {second}")
            out.append(table)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
