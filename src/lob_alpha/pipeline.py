"""The experiment pipeline.

    load -> clean -> validate -> signals -> targets -> IC -> decay -> regimes
         -> redundancy -> backtest -> robustness -> tables, figures, summary

Stages are ordered so that no trading decision is made before the statistical
question has been answered, which is the discipline the whole project is
about. Every stage writes its table to ``results/tables`` under the experiment
name, so the paper is assembled from files rather than from memory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import config_hash
from .backtest.costs import CostModel
from .backtest.engine import StrategyParams, adverse_selection, run_backtest
from .backtest.execution import ExecutionModel
from .data import cleaning, loader, sessions as session_mod
from .evaluation import decay, ic, redundancy, regimes as regime_mod
from .orderbook.representation import validate_invariants
from .robustness import bootstrap, multiple_testing, stress, walk_forward
from .signals import build_signals
from .targets.forward_returns import forward_returns, horizon_clock_time


class Results:
    """A thin namespace that writes every table it is given."""

    def __init__(self, root: Path, name: str):
        self.tables = root / "tables" / name
        self.figures = root / "figures" / name
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, str] = {}

    def table(self, key: str, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df
        path = self.tables / f"{key}.csv"
        df.to_csv(path, index=False)
        self.index[key] = str(path)
        return df

    def json(self, key: str, payload: dict) -> dict:
        path = self.tables / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        self.index[key] = str(path)
        return payload


def run(config: dict, verbose: bool = True) -> dict:
    t0 = time.time()
    config["experiment"].setdefault("config_hash", config_hash(config))
    name = config["experiment"]["name"]
    out = Results(Path(config["experiment"]["output_dir"]), name)
    log = (lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)) if verbose else (lambda msg: None)

    # -- stage 1: data ------------------------------------------------------
    log("loading data")
    raw = loader.load_raw(config["data"])
    # Sessions are assigned before cleaning, because cleaning itself looks
    # backwards (stale quotes, price jumps) and must not look across a break.
    sess_cfg = dict(config["data"]["sessions"])
    drop_short = sess_cfg.pop("drop_short", True)
    raw = session_mod.with_sessions(raw, **sess_cfg)
    book, audit = cleaning.clean(raw, **config["data"]["cleaning"])
    if drop_short:
        book, short_audit = session_mod.drop_short_sessions(book, min_events=max(config["targets"]["horizons"]) + 1)
        audit = pd.concat([audit, short_audit], ignore_index=True)
    validate_invariants(book)
    out.table("cleaning_audit", audit)
    out.table("session_report", session_mod.session_report(book))
    out.table("data_quality", cleaning.quality_report(book))
    log(f"{len(book):,} clean events, {book['instrument'].nunique()} instruments, "
        f"{session_mod.group_key(book).nunique()} sessions")

    # -- stage 2: signals and targets --------------------------------------
    log("building signals and forward returns")
    signals = build_signals(book, config["signals"])
    horizons = config["targets"]["horizons"]
    returns = forward_returns(book, horizons, use_log=config["targets"]["use_log"])
    clock = out.table("horizon_clock_time", horizon_clock_time(book, horizons))
    out.table("signal_summary", signals.describe().T.reset_index(names="signal"))

    # -- stage 3: predictive power -----------------------------------------
    ev = config["evaluation"]
    log("computing information coefficients")
    ic_long = ic.ic_table(
        signals, returns, book["instrument"], horizons,
        n_boot=ev["n_boot"], block=ev["block"], n_perm=ev["n_perm"], seed=config["experiment"]["seed"],
    )
    budget = multiple_testing.hypothesis_count(
        n_signals=len(signals.columns), n_horizons=len(horizons),
        n_regime_cells=9, n_instruments=book["instrument"].nunique(),
    )
    # Corrected over the full budget, not just the pooled table's 24 rows.
    ic_long = multiple_testing.correct_ic_table(ic_long, m=budget["total_hypotheses"])
    out.table("ic_long", ic_long)
    out.table("ic_matrix", ic.ic_matrix(ic_long).reset_index())
    out.table("ic_by_instrument", ic.ic_by_instrument(signals, returns, book["instrument"], horizons))
    # Pooled IC is dominated by whichever instrument has the most events, so the
    # cross-sectional view is reported alongside it: is the signal general, or
    # is one name carrying the whole result?
    ic_cross = out.table("ic_cross_sectional", ic.ic_cross_sectional(signals, returns, book["instrument"], horizons))
    out.table("decay_curve", decay.decay_curve(ic_long))
    out.table("half_life", decay.half_life(ic_long, clock))
    plots.plot_ic_decay(ic_long, out.figures / "ic_decay.png")

    # -- stage 4: regimes ---------------------------------------------------
    log("conditioning on regimes")
    regimes = regime_mod.assign_regimes(book, window=ev["regime_window"], vol_window=ev["vol_window"])
    ic_regime = regime_mod.ic_by_regime(signals, returns, regimes, horizons)
    out.table("ic_by_regime", ic_regime)
    out.table("regime_dependence", regime_mod.regime_dependence(ic_regime))
    for dim in regimes.columns:
        plots.plot_regime_heatmap(ic_regime, dim, out.figures / f"regime_{dim}.png")

    # -- stage 5: redundancy and shape --------------------------------------
    log("redundancy and nonlinearity")
    pear, spear = redundancy.correlation_matrices(signals)
    out.table("signal_correlation_pearson", pear.reset_index(names="signal"))
    out.table("signal_correlation_spearman", spear.reset_index(names="signal"))
    out.table("variance_inflation", redundancy.variance_inflation(signals))
    h0 = ev["headline_horizon"]
    out.table("nested_models", redundancy.nested_models(
        signals, returns[f"fwd_ret_{h0}"], order=list(signals.columns), groups=book["instrument"]))
    out.table("nested_models_order_sensitivity", redundancy.nested_models_both_orders(
        signals, returns[f"fwd_ret_{h0}"],
        collinear_pair=("queue_imbalance", "microprice_deviation"),
        base="ofi", groups=book["instrument"]))
    for sig in signals.columns:
        buckets = redundancy.quantile_response(signals[sig], returns[f"fwd_ret_{h0}"], ev["quantile_buckets"])
        if len(buckets):
            out.table(f"quantile_response_{sig}", buckets)
            plots.plot_quantile_response(buckets, sig, out.figures / f"quantile_{sig}.png")

    # -- stage 6: backtest --------------------------------------------------
    log("backtesting")
    params = StrategyParams(**config["backtest"]["strategy"])
    costs = CostModel(**config["backtest"]["costs"])
    execution = ExecutionModel(**config["backtest"]["execution"])
    run_result = run_backtest(book, signals, params, costs, execution, regimes)
    out.json("headline_metrics", run_result["metrics"])
    out.table("pnl_by_day", bootstrap.pnl_attribution(run_result["pnl"], by="day"))
    out.table("pnl_by_instrument", bootstrap.pnl_attribution(run_result["pnl"], by="instrument"))
    out.json("pnl_concentration", bootstrap.concentration(bootstrap.pnl_attribution(run_result["pnl"], by="day")))
    out.json("sharpe_confidence", bootstrap.sharpe_ci(run_result["pnl"]["net"], block=ev["block"], n_boot=ev["n_boot"]))
    plots.plot_equity_curve(run_result["pnl"], out.figures / "equity_curve.png")

    log("strategy ladder")
    out.table("strategy_comparison", strategy_ladder(book, signals, params, costs, execution, regimes))

    log("adverse selection")
    out.table(
        "adverse_selection",
        adverse_selection(book, run_result["pnl"], horizons[: min(4, len(horizons))]),
    )
    out.table(
        "adverse_selection_by_spread",
        adverse_selection(
            book, run_result["pnl"], horizons[: min(4, len(horizons))],
            conditioner=(book["ask_price"] - book["bid_price"]),
        ),
    )

    # -- stage 7: robustness ------------------------------------------------
    rb = config["robustness"]
    to_run = set(rb["run"])
    if "cost" in to_run:
        log("cost sensitivity")
        sweep = stress.cost_sensitivity(book, signals, params, costs, execution, rb["cost_multiples"], regimes)
        out.table("cost_sensitivity", sweep)
        out.json("breakeven", {"breakeven_cost_multiple": sweep.attrs.get("breakeven_cost_multiple")})
        plots.plot_cost_sensitivity(sweep, out.figures / "cost_sensitivity.png")
    if "threshold" in to_run:
        out.table("threshold_sensitivity", stress.threshold_sensitivity(book, signals, params, costs, execution, rb["thresholds"], regimes))
    if "execution" in to_run:
        out.table("execution_ladder", stress.execution_ladder(book, signals, params, costs, regimes=regimes))
    if "instrument" in to_run:
        out.table("instrument_stability", stress.instrument_stability(book, signals, params, costs, execution, regimes))
    if "period" in to_run:
        out.table("period_stability", stress.period_stability(book, signals, params, costs, execution, regimes=regimes))
    if "randomisation" in to_run:
        log("randomisation null")
        rand = stress.signal_randomisation(book, signals, params, costs, execution, rb["randomisation_draws"], ev["block"], config["experiment"]["seed"], regimes)
        out.table("signal_randomisation", rand)
        out.json("randomisation_summary", {
            "real_sharpe": rand.attrs.get("real_sharpe"),
            "real_net_pnl": rand.attrs.get("real_net_pnl"),
            "empirical_pvalue_gross": rand.attrs.get("empirical_pvalue"),
            "empirical_pvalue_net": rand.attrs.get("empirical_pvalue_net"),
            "real_sharpe_gross": rand.attrs.get("real_sharpe_gross"),
        })
        plots.plot_randomisation(rand, out.figures / "randomisation.png")
    if "subsampling" in to_run:
        out.table("subsampling_sensitivity", stress.subsampling_sensitivity(book, signals, params, costs, execution, rb["strides"], regimes))

    log("walk forward validation")
    wf_table, wf_summary = walk_forward.walk_forward(
        book, signals, params, costs, execution,
        thresholds=rb["thresholds"],
        candidate_signal_sets=[[s] for s in signals.columns] + [list(signals.columns)],
        n_folds=rb["walk_forward_folds"], purge=rb["purge"], regimes=regimes,
    )
    out.table("walk_forward", wf_table)
    out.json("walk_forward_summary", wf_summary)

    out.json("hypothesis_count", budget)
    out.json("selection_inflation", selection_inflation_study(book, signals, params, costs, execution, rb, regimes))

    # -- stage 8: summary ---------------------------------------------------
    log("per-signal economic gates")
    gates = out.table("signal_gates", signal_gates(book, signals, params, costs, execution, rb, regimes, config["experiment"]["seed"]))
    verdict = verdict_table(ic_long, wf_table, gates, signals.columns, ic_cross=ic_cross)
    out.table("verdict", verdict)
    summary_path = write_summary(config, out, run_result["metrics"], wf_summary, verdict, time.time() - t0)
    out.json("run_index", out.index)
    log(f"done in {time.time() - t0:.1f}s, summary at {summary_path}")
    return {"results": out, "verdict": verdict, "metrics": run_result["metrics"], "walk_forward": wf_summary}


def strategy_ladder(book, signals, params, costs, execution, regimes) -> pd.DataFrame:
    """A through F: does complexity actually help?"""
    cols = list(signals.columns)
    ladder = {f"{chr(65 + i)}: {c}": ([c], None) for i, c in enumerate(cols)}
    ladder[f"{chr(65 + len(cols))}: all signals"] = (cols, None)
    if "spread_regime" in regimes.columns:
        ladder[f"{chr(66 + len(cols))}: all + regime filter"] = (cols, ("spread_regime", ["medium", "high"]))
    rows = []
    for label, (sigs, filt) in ladder.items():
        p = replace(params, signals=sigs, weights=None, regime_filter=filt)
        m = run_backtest(book, signals, p, costs, execution, regimes)["metrics"]
        rows.append({
            "strategy": label, "n_signals": len(sigs),
            "gross_pnl": m["gross_pnl"], "net_pnl": m["net_pnl"], "sharpe_net": m["sharpe_net"],
            "turnover": m["turnover_total"], "max_drawdown": m["max_drawdown"],
            "hit_rate": m["hit_rate"], "cost_share_of_gross": m["cost_share_of_gross"], "exposure": m["exposure"],
        })
    return pd.DataFrame(rows)


def selection_inflation_study(book, signals, params, costs, execution, rb, regimes) -> dict:
    """Score every candidate in sample and out of sample, then measure the gap."""
    cut = int(len(book) * 0.6)
    is_scores, oos_scores = {}, {}
    for sig in signals.columns:
        for thr in rb["thresholds"]:
            p = replace(params, signals=[sig], threshold=float(thr), weights=None)
            key = f"{sig}@{thr}"
            is_scores[key] = walk_forward._slice_run(book, signals, p, costs, execution, regimes, slice(0, cut))["metrics"]["sharpe_net"]
            oos_scores[key] = walk_forward._slice_run(book, signals, p, costs, execution, regimes, slice(cut, len(book)))["metrics"]["sharpe_net"]
    return multiple_testing.selection_inflation(pd.Series(is_scores), pd.Series(oos_scores))


def signal_gates(book, signals, params, costs, execution, rb, regimes, seed: int) -> pd.DataFrame:
    """Economic gates, run once per signal rather than once per portfolio.

    The cost and randomisation verdicts belong to a signal, not to whichever
    strategy happened to be configured as the headline. Running them per signal
    is what lets the verdict table say "OFI survives costs, queue imbalance
    does not" instead of repeating one number four times.
    """
    rows = []
    for sig in signals.columns:
        p_sig = replace(params, signals=[sig], weights=None)
        sweep = stress.cost_sensitivity(book, signals, p_sig, costs, execution, rb["cost_multiples"], regimes)
        at_baseline = sweep.loc[sweep["cost_multiple"] == 1.0, "net_pnl"]
        rand = stress.signal_randomisation(
            book, signals[[sig]], p_sig, costs, execution,
            # The empirical p-value cannot fall below 1/(draws+1), so anything
            # under about 24 draws makes a 5% verdict unreachable by
            # construction rather than by evidence.
            n_draws=max(24, rb["randomisation_draws"] // 2), seed=seed, regimes=regimes,
        )
        rows.append({
            "signal": sig,
            "gross_pnl": float(sweep["gross_pnl"].iloc[0]),
            "net_pnl_at_baseline_cost": float(at_baseline.squeeze()) if len(at_baseline) else np.nan,
            "net_profitable_at_baseline_cost": bool(len(at_baseline) and float(at_baseline.squeeze()) > 0),
            "breakeven_cost_multiple": sweep.attrs.get("breakeven_cost_multiple"),
            "randomisation_pvalue_gross": rand.attrs.get("empirical_pvalue"),
            "randomisation_pvalue_net": rand.attrs.get("empirical_pvalue_net"),
            "randomisation_pvalue_floor": 1.0 / (len(rand) + 1),
            "beats_shifted_null": bool((rand.attrs.get("empirical_pvalue") or 1.0) < 0.05),
        })
    return pd.DataFrame(rows)


def verdict_table(ic_long, wf_table, gates, signal_names, ic_cross=None, min_t: float = 3.0) -> pd.DataFrame:
    """The table the whole project exists to fill in.

    Predictive -> out of sample -> net profitable -> robust. A signal has to
    clear every gate, and the honest outcome is that most do not.
    """
    rows = []
    chosen = set(wf_table["chosen_signals"].unique()) if len(wf_table) else set()
    for sig in signal_names:
        g = ic_long[ic_long["signal"] == sig]
        best = g.loc[g["rank_ic"].abs().idxmax()] if len(g) and g["rank_ic"].notna().any() else None

        # Predictive is decided cross-sectionally where more than one
        # instrument exists. A pooled rank IC ranks every instrument against
        # every other, so a signal that is strong on the two most active names
        # can dominate the pooled figure while failing on the rest. The gate
        # asks instead whether the effect holds on each instrument and whether
        # its mean is large relative to the spread across them.
        cross_row = None
        if ic_cross is not None and len(ic_cross):
            c = ic_cross[(ic_cross["signal"] == sig) & ic_cross["t_across_instruments"].notna()]
            # Best evidence across horizons, not the largest IC. A signal that
            # peaks at a long horizon usually peaks there with a wide spread
            # across instruments too, so ranking by magnitude selects the
            # noisiest cell. The question is whether the signal predicts at
            # *some* horizon, so rank by the strength of the evidence.
            if len(c):
                cross_row = c.loc[c["t_across_instruments"].abs().idxmax()]
        if cross_row is not None and cross_row["n_instruments"] > 1:
            predictive = bool(
                abs(cross_row["t_across_instruments"]) > min_t
                and cross_row["share_same_sign_as_mean"] == 1.0
            )
        else:
            predictive = bool(best is not None and best.get("ic_significant", False))
        survives_correction = bool(best is not None and bool(best.get("bh_reject", False)))
        selected_oos = any(sig in c for c in chosen)
        oos_positive = bool(len(wf_table) and (wf_table["test_sharpe"] > 0).mean() > 0.5 and selected_oos)
        rows.append({
            "signal": sig,
            "peak_abs_ic": float(best["rank_ic"]) if best is not None else np.nan,
            "peak_horizon": int(best["horizon"]) if best is not None else -1,
            "ic_mean_across_instruments": float(cross_row["ic_mean"]) if cross_row is not None else np.nan,
            "best_evidence_horizon": int(cross_row["horizon"]) if cross_row is not None else -1,
            "t_across_instruments": float(cross_row["t_across_instruments"]) if cross_row is not None else np.nan,
            "instruments_same_sign": float(cross_row["share_same_sign_as_mean"]) if cross_row is not None else np.nan,
            "predictive": predictive,
            "survives_multiple_testing": survives_correction,
            "selected_out_of_sample": selected_oos,
            "positive_out_of_sample": oos_positive,
        })
    out = pd.DataFrame(rows)
    if gates is not None and len(gates):
        out = out.merge(gates, on="signal", how="left")
    # A signal only counts as surviving if it clears every gate in sequence:
    # predictive, still predictive after correcting for how many hypotheses
    # were tested, profitable after costs, and better than its own shifted
    # null. Most will not, and that is the finding.
    out["survives_all_gates"] = (
        out["predictive"]
        & out["survives_multiple_testing"]
        & out.get("net_profitable_at_baseline_cost", False)
        & out.get("beats_shifted_null", False)
    )
    return out


def write_summary(config, out: Results, metrics, wf_summary, verdict, elapsed) -> Path:
    lines = [
        f"# {config['experiment']['name']}",
        "",
        f"Config: `{config['experiment'].get('config_path')}` (hash `{config['experiment']['config_hash']}`)  ",
        f"Runtime: {elapsed:.1f}s",
        "",
        "## Headline backtest",
        "",
        f"- signals: {metrics['signals']}, threshold {metrics['threshold']}, execution {metrics['execution']}",
        f"- gross P&L {metrics['gross_pnl']:.4f}, net P&L {metrics['net_pnl']:.4f}",
        f"- costs took {100 * (metrics['cost_share_of_gross'] or float('nan')):.1f}% of gross",
        f"- net Sharpe {metrics['sharpe_net']:.4f} per event against gross {metrics['sharpe_gross']:.4f} per event",
        f"- annualised at sqrt of time that is {metrics['sharpe_net_annualised']:.1f} net, which the block bootstrap should be read against rather than quoted alone",
        f"- {metrics['n_trades']:,} trades, exposure {100 * metrics['exposure']:.1f}%, max drawdown {metrics['max_drawdown']:.4f}",
        "",
        "## Walk forward",
        "",
        f"- {wf_summary['n_folds']} folds, {wf_summary['folds_positive_test']} with positive test Sharpe",
        f"- mean test Sharpe {wf_summary['mean_test_sharpe']:.4f} against mean train Sharpe {wf_summary['mean_train_sharpe']:.4f} (per event)",
        f"- in-sample inflation {wf_summary['in_sample_inflation']:.4f} Sharpe points per event",
        "",
        "## Verdict",
        "",
        verdict.to_markdown(index=False),
        "",
        "Tables in `" + str(out.tables) + "`, figures in `" + str(out.figures) + "`.",
        "",
    ]
    path = out.tables.parent.parent / f"summary_{config['experiment']['name']}.md"
    path.write_text("\n".join(lines))
    return path
