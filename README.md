Limit Order Book: Alpha Research Framework

A research framework for testing whether limit order book state predicts short-horizon price moves, and whether any of that prediction survives realistic execution costs.

Companion code for the paper:

> **When Does Limit Order Book Information Constitute Tradable Alpha?**
> William Odumosu, August 2026. 
> Five NASDAQ instruments, 953,829 events. Three of four signals are genuinely predictive on every instrument. None survives the spread.

## What this does

Most limit order book projects report a rank IC, a Sharpe ratio, and a rising equity curve, without ever separating a measurement from an edge. This framework keeps them apart.

Given tick or snapshot order book data, it:

- Reconstructs a canonical book representation, assigns trading sessions, and validates its invariants before any research happens
- Computes four microstructure signals: order flow imbalance, queue imbalance, micro-price deviation, and spread
- Measures rank IC across six forward horizons, with block bootstrap intervals and block permutation p-values
- Maps signal decay, half-lives, and dependence on spread, volatility, and liquidity regimes
- Tests signal redundancy via correlation, variance inflation, and nested models scored out of sample
- Backtests under an explicit, parameterised cost model and a five-rung execution ladder
- Attacks its own result: walk-forward validation, multiple-testing correction, selection-inflation measurement, and block-shifted randomisation nulls

## Setup

```
pip install -e ".[dev]"
```

Requires Python 3.10 or later. Dependencies are numpy, pandas, scipy, matplotlib, and pyyaml.

## Usage

### Reproduce the paper

```
python run_experiment.py configs/lobster_panel.yaml
python experiments/finish_gates.py configs/lobster_panel.yaml
```

Produces every table and figure in the paper: 46 result files in `results/tables/lobster_panel/`, 11 figures, and a markdown summary. Roughly five minutes total on 953,829 events, split across two commands because the per-signal gate stage is over a hundred backtests on its own. Requires the LOBSTER samples in `data/raw/`; see `data/README.md`.

### Run on synthetic data, no download needed

```
python run_experiment.py configs/baseline.yaml
```

A planted-signal market that ships with the repository, so the pipeline runs end to end with no data licence. Roughly 40 seconds on 120,000 events.

### Run the synthetic controls

```
python run_experiment.py configs/null_market.yaml
python run_experiment.py configs/regime_market.yaml
```

The null market has no relationship by construction, so a positive result there is a bug in the pipeline rather than an edge. The regime market plants a relationship active only in wide spreads, so the regime analysis has something known to recover.

### Override any parameter from the command line

```
python run_experiment.py configs/baseline.yaml --set backtest.strategy.threshold=1.5
```

### Point it at LOBSTER data

```yaml
data:
  source: lobster
  path: data/raw/
  levels: 1
```

Files are paired by the LOBSTER naming convention, so a directory of several tickers or several days loads as one multi-instrument, multi-session panel.

### Point it at any other data

```yaml
data:
  source: file
  path: data/raw/my_book/
  column_map: {ts: timestamp, symbol: instrument, bid: bid_price}
```

Nothing downstream changes. The canonical schema is `timestamp, instrument, bid_price, bid_size, ask_price, ask_size`, with optional deeper levels as `bid_price_2` and so on.

### Run the tests

```
pytest
```

80 tests covering book invariants, every branch of the OFI definition by hand, cross-instrument leakage, cost and backtest accounting identities, split geometry, and recovery of planted relationships from synthetic data.

## Example output

```
$ python run_experiment.py configs/lobster_panel.yaml

[   9.2s] 953,829 clean events, 5 instruments, 5 sessions
[ 140.9s] conditioning on regimes
[ 156.5s] backtesting
[ 218.1s] per-signal economic gates
[ 286.2s] done in 286.2s, summary at results/summary_lobster_panel.md
```

The verdict table is the point of the whole repository. Five NASDAQ names, 21 June 2012:

```
signal                 mean_ic    t_xsec  predictive  correction  net_profitable  beats_null
ofi                      0.164     17.99        True        True           False        True
queue_imbalance          0.150      7.32        True        True           False        True
microprice_deviation     0.105      7.56        True        True           False        True
spread                   0.005      1.07       False       False           False       False
```

Three of four signals carry genuine information, holding the same sign on every instrument. None survives the spread. Profitability dies at 0.048 times the baseline cost assumption:

```
cost_multiple    gross_pnl     net_pnl   sharpe_net   cost_share_of_gross
         0.00       0.4217      0.4217       0.0317                  0.00
         0.25       0.4217     -1.7912      -0.1029                  5.25
         0.50       0.4217     -4.0040      -0.1522                 10.50
         1.00       0.4217     -8.4296      -0.1773                 20.99
```

Which signal wins depends on the book. OFI dominates the wide-spread, thin-queue names; queue imbalance dominates the penny-spread, deep-queue ones, and the ordering flips completely between them:

```
instrument      ofi  queue_imbalance  microprice_deviation   spread
AAPL          0.181            0.063                 0.054    0.026
AMZN          0.222            0.151                 0.145    0.037
GOOG          0.183            0.113                 0.106   -0.019
INTC          0.108            0.228                 0.220   -0.009
MSFT          0.114            0.263                 0.257   -0.010
```

## How it works

**Canonical representation**: Raw vendor columns are mapped onto one schema, then validated for crossed books, locked books, non-positive sizes, and per-instrument timestamp monotonicity. Every downstream module is entitled to assume those invariants hold.

**Signals**: OFI uses the Cont, Kukanov, and Stoikov construction, which conditions on the price move so that a queue growing from arrivals, shrinking from cancellations, and vanishing from a price level move are counted differently. The other three are top-of-book quantities. All four are frozen at the protocol stage, so adding a fifth requires a visible edit to the registry rather than an invisible one in a notebook.

**Targets**: Forward mid returns over 1, 5, 10, 25, 50, and 100 events, with every horizon also reported in elapsed seconds. Fifty events in one regime is not fifty events in another.

**Headline metric**: Rank IC, because it is invariant to monotone rescaling and less hostage to fat-tailed high-frequency returns than Pearson. Intervals come from a circular block bootstrap and p-values from a block permutation, both of which respect serial dependence that ordinary standard errors ignore.

**Costs**: Parameterised, not calibrated. A market impact coefficient estimated from public data is a guess wearing a lab coat, so the honest move is to state the functional form, sweep the coefficients, and report where profitability flips.

**Sessions**: An instrument is not a contiguous series. Every shift, roll, and diff groups by instrument and session, so a 100-event forward return at the close returns missing rather than an overnight return mislabelled as a horizon. Positions start flat each session and earn nothing across the gap.

**Look-ahead prevention**: Structural, not disciplinary. Latency lives in the execution model, so a strategy that trades on information it did not have cannot be expressed. Z-scores and regime labels use trailing windows only, and one test truncates the future to assert that no past position changes.

**Validation**: Expanding-window splits with a purge gap, selecting on validation and touching test once per fold. The selection-inflation study scores every candidate specification both in sample and out of sample, so the gap is measured rather than assumed away.

## File structure

```
run_experiment.py              CLI entry point
configs/                       baseline, null_market, regime_market, cost_sensitivity, robustness
src/lob_alpha/
    config.py                  one yaml fully determines an experiment
    pipeline.py                stage ordering, tables, figures, summary
    plots.py                   every figure regenerates from a written table
    data/                      loader, cleaning, sessions, synthetic controls
    orderbook/                 canonical schema and its invariants
    signals/                   ofi, queue_imbalance, microprice, spread, registry
    targets/                   forward returns in event and clock time
    evaluation/                ic, decay, regimes, redundancy, significance
    backtest/                  engine, execution ladder, cost model
    robustness/                walk_forward, bootstrap, multiple_testing, stress
tests/                         80 tests, including synthetic recovery
docs/protocol.md               the frozen research protocol and known limitations
notebooks/                     exploration only, never a reported result
```

## Reproducibility

Raw data is read-only. Cleaning writes to `data/processed/` and records every dropped row with its reason. Each config carries a content hash, written into the summary. Notebooks are for exploration: once an experiment is intended for reporting it moves into the pipeline, which is what protects the results from being quietly re-tuned.

## Roadmap

- Queue-aware passive execution (level 5), currently declared and raising `NotImplementedError` rather than faked.
- Probability forecasting scored on log loss, Brier score, and calibration rather than on returns.
- Nonlinear modelling, last. "Can gradient boosting extract incremental information from already validated signals" is a far stronger question than "what happens if I throw XGBoost at an order book".

## Licence

MIT.

Thanks for reading :)
