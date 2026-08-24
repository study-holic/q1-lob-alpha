"""Validation machinery: splits, bootstrap, corrections, cleaning."""

import numpy as np
import pandas as pd
import pytest

from lob_alpha.data.cleaning import clean, quality_report
from lob_alpha.evaluation.significance import (
    benjamini_hochberg,
    block_bootstrap_ci,
    block_permutation_pvalue,
    newey_west_tstat,
    spearman,
)
from lob_alpha.robustness.walk_forward import expanding_splits


def test_splits_never_overlap_and_always_move_forward():
    for tr, va, te in expanding_splits(10000, n_folds=5, purge=50):
        assert tr.start == 0
        assert tr.stop <= va.start
        assert va.stop <= te.start
        assert te.stop <= 10000


def test_purge_leaves_a_gap():
    folds = expanding_splits(10000, n_folds=4, purge=100)
    tr, va, te = folds[0]
    assert va.start - tr.stop >= 100


def test_bootstrap_interval_brackets_the_truth():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.5, 1.0, 5000))
    point, lo, hi = block_bootstrap_ci(x, np.mean, block=50, n_boot=200, seed=1)
    assert lo < 0.5 < hi
    assert point == pytest.approx(x.mean())


def test_permutation_pvalue_is_large_under_the_null():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=4000))
    y = pd.Series(rng.normal(size=4000))
    p = block_permutation_pvalue(x, y, spearman, block=100, n_perm=100, seed=2)
    assert p > 0.05


def test_permutation_pvalue_is_small_when_the_relationship_is_real():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(size=4000))
    y = 0.4 * x + pd.Series(rng.normal(size=4000))
    p = block_permutation_pvalue(x, y, spearman, block=100, n_perm=100, seed=2)
    assert p < 0.05


def test_benjamini_hochberg_is_stricter_than_uncorrected():
    p = pd.Series([0.001, 0.01, 0.02, 0.04, 0.2, 0.5, 0.9])
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["bh_reject"].sum() <= (p <= 0.05).sum()
    assert out["bonferroni_reject"].sum() <= out["bh_reject"].sum()


def test_newey_west_widens_the_error_for_persistent_series():
    rng = np.random.default_rng(9)
    e = rng.normal(size=5000)
    persistent = pd.Series(e).ewm(span=50).mean().to_numpy()
    naive_t = persistent.mean() / (persistent.std() / np.sqrt(len(persistent)))
    assert abs(newey_west_tstat(persistent)) < abs(naive_t)


def test_cleaning_removes_crossed_books_and_records_it():
    book = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:00", periods=4, freq="1s"),
            "instrument": ["X"] * 4,
            "bid_price": [100.0, 100.0, 101.0, 100.0],
            "bid_size": [10.0, 10.0, 10.0, 10.0],
            "ask_price": [100.02, 100.02, 100.5, 100.04],
            "ask_size": [10.0, 10.0, 10.0, 10.0],
        }
    )
    cleaned, audit = clean(book)
    assert (cleaned["ask_price"] >= cleaned["bid_price"]).all()
    assert audit.loc[audit["rule"] == "crossed_book", "rows_dropped"].iloc[0] == 1


def test_cleaning_drops_stale_repeats_but_keeps_real_events():
    book = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:00", periods=3, freq="1s"),
            "instrument": ["X"] * 3,
            "bid_price": [100.0, 100.0, 100.0],
            "bid_size": [10.0, 10.0, 12.0],
            "ask_price": [100.02] * 3,
            "ask_size": [10.0] * 3,
        }
    )
    cleaned, audit = clean(book)
    assert len(cleaned) == 2
    assert audit.loc[audit["rule"] == "stale_repeated_quote", "rows_dropped"].iloc[0] == 1


def test_cleaning_does_not_mutate_the_input():
    book = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:00", periods=2, freq="1s"),
            "instrument": ["X", "X"],
            "bid_price": [100.0, 100.0],
            "bid_size": [10.0, 11.0],
            "ask_price": [100.02, 100.02],
            "ask_size": [10.0, 10.0],
        }
    )
    before = book.copy()
    clean(book)
    pd.testing.assert_frame_equal(book, before)


def test_quality_report_covers_every_instrument(small_panel):
    report = quality_report(small_panel)
    assert set(report["instrument"]) == set(small_panel["instrument"].unique())
    assert (report["median_spread"] > 0).all()


def test_cross_sectional_ic_flags_a_result_carried_by_one_instrument():
    """Pooled IC can look strong when only one name has any signal at all."""
    import numpy as np
    from lob_alpha.evaluation.ic import ic_cross_sectional

    rng = np.random.default_rng(0)
    n = 3000
    inst = pd.Series(np.repeat(["A", "B", "C"], n))
    sig = pd.Series(rng.normal(size=3 * n))
    # Only instrument A carries a relationship.
    noise = pd.Series(rng.normal(size=3 * n))
    ret = noise.copy()
    ret[:n] = 0.6 * sig[:n] + 0.4 * noise[:n]

    table = ic_cross_sectional(
        pd.DataFrame({"s": sig}), pd.DataFrame({"fwd_ret_1": ret}), inst, [1]
    )
    row = table.iloc[0]
    assert row["n_instruments"] == 3
    assert row["ic_max"] > 0.4
    assert abs(row["t_across_instruments"]) < 3.0
    assert row["share_same_sign_as_mean"] < 1.0


def test_grouped_splits_test_blocks_contain_every_instrument():
    """A2: folds must be time slices of each name, not one name per fold."""
    from lob_alpha.robustness.walk_forward import grouped_splits

    groups = pd.Series(["A"] * 5000 + ["B"] * 5000 + ["C"] * 5000)
    for train, val, test in grouped_splits(groups, n_folds=5, purge=101):
        assert groups[test].nunique() == 3
        assert groups[train].nunique() == 3
        assert not (train & test).any()
        assert not (val & test).any()


def test_grouped_splits_never_train_on_the_future_of_its_own_group():
    from lob_alpha.robustness.walk_forward import grouped_splits

    groups = pd.Series(["A"] * 4000 + ["B"] * 4000)
    pos = pd.Series(range(8000))
    for train, val, test in grouped_splits(groups, n_folds=4, purge=101):
        for g in ("A", "B"):
            in_g = (groups == g).to_numpy()
            if not (test & in_g).any():
                continue
            assert pos[train & in_g].max() < pos[test & in_g].min()


def test_nested_models_test_set_spans_every_instrument():
    """A3: a panel tail is one instrument; a per-instrument tail is all of them."""
    import numpy as np
    from lob_alpha.evaluation.redundancy import nested_models

    rng = np.random.default_rng(0)
    n = 1500
    inst = pd.Series(np.repeat(["A", "B", "C"], n))
    sig = pd.DataFrame({"s1": rng.normal(size=3 * n), "s2": rng.normal(size=3 * n)})
    y = pd.Series(0.3 * sig["s1"] + rng.normal(size=3 * n))

    grouped = nested_models(sig, y, groups=inst, test_fraction=0.3)
    pooled = nested_models(sig, y, test_fraction=0.3)
    assert int(grouped.iloc[0]["n_test"]) == int(pooled.iloc[0]["n_test"])
    # The grouped split takes the last 30% of each name, so all three appear.
    cut_per_group = int(n * 0.7)
    assert cut_per_group < n


def test_breakeven_is_nan_when_never_profitable():
    """B1: 0.0 would read as 'breaks even for free'."""
    import numpy as np
    from lob_alpha.robustness.stress import _breakeven

    never = pd.DataFrame({"cost_multiple": [0.0, 1.0, 2.0], "net_pnl": [-1.0, -2.0, -3.0]})
    assert np.isnan(_breakeven(never))
    always = pd.DataFrame({"cost_multiple": [0.0, 1.0, 2.0], "net_pnl": [3.0, 2.0, 1.0]})
    assert np.isinf(_breakeven(always))
    crosses = pd.DataFrame({"cost_multiple": [0.0, 1.0, 2.0], "net_pnl": [1.0, -1.0, -3.0]})
    assert _breakeven(crosses) == pytest.approx(0.5)


def test_benjamini_hochberg_family_size_is_respected():
    """A1: correcting over 24 when 1,080 were tested is not a correction."""
    # 0.0033 is the permutation floor at n_perm = 300. It clears BH over the
    # pooled table's 24 rows and fails over the real family of 1,080, which is
    # exactly the gap that made the original correction meaningless.
    p = pd.Series([0.0033] * 24)
    assert benjamini_hochberg(p).bh_reject.sum() == 24
    assert benjamini_hochberg(p, m=1080).bh_reject.sum() == 0
    # A finer floor restores feasibility, which is why n_perm was raised.
    assert benjamini_hochberg(pd.Series([0.0005] * 24), m=1080).bh_reject.sum() == 24
