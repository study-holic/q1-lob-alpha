"""Recompute the per-signal gates, the verdict, and the summary.

The gates stage runs a cost sweep and a randomisation null for every signal
separately, which is over a hundred full backtests. On a large panel that is
the most expensive part of the pipeline by a wide margin, and it depends on
nothing that the earlier stages compute beyond the book, the signals, and the
regime labels.

Splitting it out lets a long panel run finish in two bounded steps instead of
one unbounded one. It reads and rewrites exactly the three outputs it owns:
``signal_gates``, ``verdict``, and the run summary.

    python run_experiment.py configs/lobster_panel.yaml
    python experiments/finish_gates.py configs/lobster_panel.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lob_alpha.backtest.costs import CostModel  # noqa: E402
from lob_alpha.backtest.engine import StrategyParams  # noqa: E402
from lob_alpha.backtest.execution import ExecutionModel  # noqa: E402
from lob_alpha.config import load_config  # noqa: E402
from lob_alpha.data import cleaning, loader, sessions as session_mod  # noqa: E402
from lob_alpha.evaluation import regimes as regime_mod  # noqa: E402
from lob_alpha.pipeline import Results, signal_gates, verdict_table, write_summary  # noqa: E402
from lob_alpha.signals import build_signals  # noqa: E402


def main() -> int:
    t0 = time.time()
    cfg = load_config(sys.argv[1])
    name = cfg["experiment"]["name"]
    out = Results(Path(cfg["experiment"]["output_dir"]), name)

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    log("loading data")
    sess = dict(cfg["data"]["sessions"])
    sess.pop("drop_short", True)
    raw = session_mod.with_sessions(loader.load_raw(cfg["data"]), **sess)
    book, _ = cleaning.clean(raw, **cfg["data"]["cleaning"])
    book, _ = session_mod.drop_short_sessions(book, min_events=max(cfg["targets"]["horizons"]) + 1)
    book = book.reset_index(drop=True)

    signals = build_signals(book, cfg["signals"])
    ev = cfg["evaluation"]
    regimes = regime_mod.assign_regimes(book, window=ev["regime_window"], vol_window=ev["vol_window"])

    params = StrategyParams(**cfg["backtest"]["strategy"])
    costs = CostModel(**cfg["backtest"]["costs"])
    execution = ExecutionModel(**cfg["backtest"]["execution"])

    log("per-signal economic gates")
    gates = out.table(
        "signal_gates",
        signal_gates(book, signals, params, costs, execution, cfg["robustness"], regimes, cfg["experiment"]["seed"]),
    )

    tables = out.tables
    ic_long = pd.read_csv(tables / "ic_long.csv")
    ic_cross = pd.read_csv(tables / "ic_cross_sectional.csv")
    wf = pd.read_csv(tables / "walk_forward.csv")
    metrics = pd.read_json(tables / "headline_metrics.json", typ="series").to_dict()
    wf_summary = pd.read_json(tables / "walk_forward_summary.json", typ="series").to_dict()

    verdict = out.table("verdict", verdict_table(ic_long, wf, gates, signals.columns, ic_cross=ic_cross))
    path = write_summary(cfg, out, metrics, wf_summary, verdict, time.time() - t0)
    log(f"done, summary at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
