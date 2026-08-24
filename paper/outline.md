# Paper outline

Target length 10 to 20 pages. Each section names the table or figure that fills it, so writing is assembly rather than invention. Nothing here is written until the experiment that fills it has run.

## 1. Abstract (150 to 250 words)
State the question, the dataset, the headline rank IC, the cost multiple at which profitability disappears, and which signals cleared every gate.

## 2. Introduction
Why short-horizon prediction matters, and why statistical significance is a weak claim in this setting. Ends with the central distinction the paper defends: correlation, prediction, out-of-sample prediction, and tradable alpha are four different things.

## 3. Data
`data_quality.csv`, `cleaning_audit.csv`, `horizon_clock_time.csv`. Instrument coverage, sampling, session handling, and every cleaning rule with its row count.

## 4. Market microstructure signals
Mathematical definitions of OFI, queue imbalance, micro-price deviation, and spread. The OFI derivation is given in full, including the three-case reading of the indicator terms. Unit tests for each branch are cited in the appendix.

## 5. Experimental methodology
Targets, horizons in both units, rank IC as the headline metric, block bootstrap intervals, block permutation tests, and the walk-forward geometry with its purge gap. States the frozen protocol and the hypothesis count.

## 6. Predictive results
`ic_matrix.csv`, `ic_long.csv`, `ic_by_instrument.csv`, figure `ic_decay.png`. The signal by horizon matrix, then decay curves and half-lives in events and seconds.

## 7. Regime dependence
`ic_by_regime.csv`, `regime_dependence.csv`, figures `regime_*.png`. Whether a signal is weak overall but strong in one regime, or predictive only because it proxies for the regime itself.

## 8. Signal redundancy
`signal_correlation_*.csv`, `variance_inflation.csv`, `nested_models.csv`, figures `quantile_*.png`. Incremental out-of-sample R squared, which can and does go negative. Quantile response curves for non-monotone structure.

## 9. Economic significance
`headline_metrics.json`, `strategy_comparison.csv`, `cost_sensitivity.csv`, `execution_ladder.csv`, `adverse_selection*.csv`, figures `cost_sensitivity.png`, `equity_curve.png`. The strategy ladder answers whether complexity helps. The adverse selection section answers whether profit comes from providing liquidity or from being picked off slowly.

## 10. Robustness
`walk_forward.csv`, `signal_randomisation.csv`, `selection_inflation.json`, `hypothesis_count.json`, `period_stability.csv`, `instrument_stability.csv`, `subsampling_sensitivity.csv`, figure `randomisation.png`. Structured as an attempt to destroy the result, not to defend it.

## 11. Failure analysis
Where each signal stops working, and the specific assumption that carries it. The cost multiple at which a positive backtest turns negative belongs here as a single sentence with a number in it.

## 12. Conclusion
`verdict.csv`. Which signals cleared predictive, correction, cost, and null-beating gates, and what the ones that failed reveal about the difference between a measurement and an edge.

## Appendix
Full OFI derivation, cleaning rules, test inventory, synthetic control specifications, and the known limitations from `docs/protocol.md`.
