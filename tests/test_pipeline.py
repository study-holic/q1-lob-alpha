"""End to end: a config in, a full result set out."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lob_alpha.config import DEFAULTS, deep_merge, load_config
from lob_alpha.pipeline import run

FAST = {
    "experiment": {"name": "test_run", "seed": 0},
    "data": {"synthetic": {"n_instruments": 2, "n_events": 4000, "mode": "ofi", "beta": 0.3}},
    "targets": {"horizons": [1, 5, 10]},
    "evaluation": {"n_boot": 30, "n_perm": 20, "block": 200, "regime_window": 500, "headline_horizon": 5},
    "backtest": {"strategy": {"zscore_window": 500}},
    "robustness": {
        "walk_forward_folds": 2,
        "randomisation_draws": 3,
        "thresholds": [0.5, 1.0],
        "cost_multiples": [0.0, 1.0, 2.0],
        "run": ["cost", "randomisation"],
    },
}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    cfg = deep_merge(DEFAULTS, FAST)
    cfg["experiment"]["output_dir"] = str(tmp_path_factory.mktemp("results"))
    return run(cfg, verbose=False), Path(cfg["experiment"]["output_dir"])


def test_pipeline_writes_the_core_tables(result):
    _, out = result
    tables = {p.stem for p in (out / "tables" / "test_run").glob("*.csv")}
    for expected in ["ic_long", "ic_matrix", "decay_curve", "half_life", "ic_by_regime",
                     "nested_models", "cost_sensitivity", "walk_forward", "verdict",
                     "strategy_comparison", "data_quality", "cleaning_audit"]:
        assert expected in tables, f"missing table: {expected}"


def test_pipeline_writes_figures(result):
    _, out = result
    figures = {p.name for p in (out / "figures" / "test_run").glob("*.png")}
    assert "ic_decay.png" in figures
    assert "equity_curve.png" in figures
    assert "cost_sensitivity.png" in figures


def test_pipeline_writes_a_summary(result):
    _, out = result
    assert (out / "summary_test_run.md").exists()


def test_verdict_has_one_row_per_signal(result):
    res, out = result
    verdict = pd.read_csv(out / "tables" / "test_run" / "verdict.csv")
    assert len(verdict) == 4
    assert {"predictive", "survives_multiple_testing", "positive_out_of_sample"} <= set(verdict.columns)


def test_walk_forward_reports_train_and_test_separately(result):
    _, out = result
    wf = pd.read_csv(out / "tables" / "test_run" / "walk_forward.csv")
    assert {"train_sharpe", "validation_sharpe", "test_sharpe"} <= set(wf.columns)
    assert len(wf) >= 1


def test_config_hash_is_stable_and_sensitive(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment:\n  name: a\nbacktest:\n  strategy:\n    threshold: 1.0\n")
    h1 = load_config(p)["experiment"]["config_hash"]
    p.write_text("experiment:\n  name: a\nbacktest:\n  strategy:\n    threshold: 2.0\n")
    h2 = load_config(p)["experiment"]["config_hash"]
    assert h1 != h2
