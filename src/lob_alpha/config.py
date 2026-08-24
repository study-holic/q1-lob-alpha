"""Configuration.

One yaml file fully determines an experiment. A result that cannot be produced
by ``python run_experiment.py configs/<something>.yaml`` is not a result, it is
a notebook cell.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "experiment": {"name": "baseline", "seed": 0, "output_dir": "results"},
    "data": {
        "source": "synthetic",
        "synthetic": {"n_instruments": 4, "n_events": 30000, "n_sessions": 5, "mode": "ofi", "beta": 0.12, "seed": 0},
        "cleaning": {"drop_locked": True, "max_relative_jump": 0.05},
        "sessions": {"method": "calendar_day", "max_gap_seconds": 1800, "drop_short": True},
    },
    "signals": ["ofi", "queue_imbalance", "microprice_deviation", "spread"],
    "targets": {"horizons": [1, 5, 10, 25, 50, 100], "use_log": False},
    "evaluation": {
        "n_boot": 300,
        "block": 500,
        "n_perm": 500,  # sampled-fallback only; the IC shift test is exact
        "regime_window": 2000,
        "vol_window": 200,
        "quantile_buckets": 10,
        "headline_horizon": 10,
    },
    "backtest": {
        "strategy": {"signals": ["ofi"], "threshold": 1.0, "zscore_window": 2000, "position_limit": 1.0},
        "costs": {"fixed_bps": 0.0, "half_spread_multiplier": 1.0, "impact_coefficient": 0.0, "participation": 0.01},
        "execution": {"level": 2, "slippage_fraction_of_spread": 0.25, "latency_events": 1},
    },
    "robustness": {
        "cost_multiples": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        "thresholds": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        "walk_forward_folds": 5,
        "purge": 101,  # one more than the longest horizon, so no return lands on a boundary
        "randomisation_draws": 50,
        "strides": [1, 2, 5],
        "run": ["cost", "threshold", "execution", "instrument", "period", "randomisation", "subsampling"],
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULTS, user)
    cfg["experiment"]["config_path"] = str(path)
    cfg["experiment"]["config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    payload = {k: v for k, v in cfg.items() if k != "experiment"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
