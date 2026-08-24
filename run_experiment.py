#!/usr/bin/env python
"""Entry point. One config in, one reproducible set of results out.

    python run_experiment.py configs/baseline.yaml
    python run_experiment.py configs/baseline.yaml --set data.synthetic.n_events=5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from lob_alpha.config import load_config  # noqa: E402
from lob_alpha.pipeline import run  # noqa: E402


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = _coerce(value)
    return cfg


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",")]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one LOB alpha experiment")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append",
                        help="dotted override, e.g. --set backtest.strategy.threshold=1.5")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    run(cfg, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
