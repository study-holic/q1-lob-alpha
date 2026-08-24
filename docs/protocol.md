# Research protocol

## Primary question

How much incremental information about short-horizon price movements is contained in limit order book state, and how much of that information remains after realistic execution costs?

## Secondary questions

1. Which of the four signals has the strongest predictive power?
2. How quickly does each signal decay?
3. Does predictive power depend on the spread, volatility, or liquidity regime?
4. Does a statistically significant signal remain economically useful?
5. How sensitive are the results to transaction cost assumptions?
6. How much does repeated signal selection inflate apparent performance?
7. Do the signals hold on genuinely unseen data?
8. Are the signals redundant, or complementary?

## Signal set

- **Order flow imbalance (OFI)**: the Cont, Kukanov, and Stoikov construction, rolling over a fixed event window.
- **Queue imbalance**: normalised top-of-book size difference.
- **Micro-price deviation**: size-weighted mid minus mid, in relative terms.
- **Spread**: relative spread, tested as a directional predictor and used as a conditioning variable.

A fifth signal requires an edit to `signals/__init__.py`, which records the decision in version control.

## Targets

Forward mid returns over horizons of 1, 5, 10, 25, 50, and 100 events, with every horizon also reported in elapsed seconds. Event count and clock time diverge across regimes, so a decay result needs both units to be interpretable.

## Headline metric

Rank IC, the Spearman correlation between the signal at time `t` and the forward return over the next `h` events. Rank IC is invariant to monotone rescaling of the signal, and its reliance on ranks limits the influence of the fat tails present in high-frequency returns.

## Validation rules

- **Every backward or forward looking operation groups by instrument and session.** Shifts, rolls, diffs, and position carry all respect session boundaries. Grouping by instrument alone converts an overnight gap into an h-event return.
- **Rank IC is reported pooled and cross-sectionally.** The pooled figure is weighted by event count and dominated by the most active instrument. The cross-sectional standard error measures whether the signal is general.
- **No random shuffling of time series observations.** Splits are expanding-window and chronological.
- **Splits are taken within each instrument, never by row position on a panel.** A panel is stored instrument-major, so slicing the concatenated frame assigns each fold's test set to a single ticker and converts a temporal holdout into a cross-sectional one.
- **Conditioning variables are ranked within instrument.** An absolute spread quintile on a mixed panel sorts tickers, and gives no information about market state.
- **Purge gap between train and test.** No overlapping forward return may straddle a split boundary.
- **Selection happens on validation, evaluation happens once on test.** The test slice has no influence on any choice.
- **Regime labels and z-scores use trailing windows.** A label assigned at time `t` must be computable at time `t`.
- **Latency is enforced in the execution model.** A signal observed at `t` trades on the book at `t + 1` at the earliest, which makes a look-ahead strategy inexpressible.

## Success criteria

1. A short-horizon edge that holds after costs, out-of-sample testing, and multiple-testing correction.
2. A demonstration that apparent alpha disappears under realistic costs and honest validation, with the specific point of failure identified.

## Known limitations

- **The circular-shift permutation null is mildly anti-conservative.** On the synthetic null control its distribution is roughly 40 per cent narrower than the spread of rank IC across independent realisations, since shifting preserves each series' own autocorrelation and not the variation between realised paths. Single-sample p-values should be read as optimistic, and the multi-seed control in `tests/test_synthetic_recovery.py` is the stronger check.
- **The IC shift test is exact; the gate-level randomisation null is sampled.** The IC table enumerates every admissible circular shift by cross-correlation and carries no sampling floor. The per-signal gates sample 24 shifted runs, giving a p-value floor of 0.04 for every signal.
- **Multiple-testing correction is taken at the full family size of 1,080**, not at the size of the displayed table. A correction over the 24 rows shown accounts for a fraction of the specifications evaluated. When the shift test was sampled at 300 draws, its floor of 0.0033 exceeded the rank-24 threshold of 0.0011, and no rejection at the true family size was possible for any data.
- **Rank IC bootstraps resample pre-computed ranks.** Ranks are not recomputed inside each draw. The approximation keeps several hundred bootstrap draws affordable on a full panel.
- **Annualised Sharpe assumes independence across events.** The block bootstrap exists to test that assumption, so per-event Sharpe is the primary figure throughout.

## Amendments

**August 2026, post-audit.** Four defects were found in the implementation, each of which this protocol had asserted was handled. Walk-forward folds and the nested-model split were taken by row position on an instrument-major panel, so each test block held a single ticker. The adverse-selection conditioner was ranked across the pooled panel, which made it a proxy for instrument identity. Benjamini-Hochberg was applied at the size of the displayed table in place of the stated family of 1,080. All four are fixed, all four are documented in the paper, and the results that changed are recorded there.
