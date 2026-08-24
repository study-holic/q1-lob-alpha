# Research protocol

Frozen before the first result was looked at. Amendments go at the bottom, dated, with a reason. Editing the questions after seeing the answers is how a research project turns into a fishing expedition.

## Primary question

How much incremental information about short-horizon price movements is contained in limit order book state, and how much of that information remains after realistic execution costs?

## Secondary questions

1. Which of the four signals has the strongest predictive power?
2. How quickly does each signal decay?
3. Does predictive power depend on the spread, volatility, or liquidity regime?
4. Does a statistically significant signal remain economically useful?
5. How sensitive are the results to transaction cost assumptions?
6. How much does repeated signal selection inflate apparent performance?
7. Do the signals survive genuinely unseen data?
8. Are the signals redundant, or genuinely complementary?

## Signal set (frozen)

Four signals, no more, until every question above has an answer:

- **Order flow imbalance (OFI)**: the Cont, Kukanov, and Stoikov construction, rolling over a fixed event window.
- **Queue imbalance**: normalised top-of-book size difference.
- **Micro-price deviation**: size-weighted mid minus mid, in relative terms.
- **Spread**: relative spread, tested as a directional predictor and used as a conditioning variable.

Adding a fifth signal requires editing `signals/__init__.py`, which makes the decision visible in version control rather than invisible in a notebook.

## Targets

Forward mid returns over horizons of 1, 5, 10, 25, 50, and 100 events, with every horizon also reported in elapsed seconds. Event count and clock time diverge across regimes, so a claim about decay is meaningless in only one of the two units.

## Headline metric

Rank IC (Spearman correlation between the signal at time `t` and the forward return over the next `h` events). It is invariant to monotone rescaling of the signal, and far less hostage to fat-tailed high-frequency returns than Pearson.

## Validation rules

- **Every backward or forward looking operation groups by instrument and session.** Shifts, rolls, diffs, and position carry all respect session boundaries. Grouping by instrument alone silently converts an overnight gap into an h-event return.
- **Rank IC is reported pooled and cross-sectionally.** The pooled figure is dominated by the most active instrument; the cross-sectional standard error answers whether the signal is general.
- **No random shuffling of time series observations, ever.** Splits are expanding-window and chronological.
- **Splits are taken within each instrument, never by row position on a panel.** A panel is stored instrument-major, so slicing the concatenated frame hands each fold's test set to a single ticker and silently converts a temporal holdout into a cross-sectional one.
- **Conditioning variables are ranked within instrument.** An absolute spread quintile on a mixed panel sorts tickers, not market states.
- **Purge gap between train and test.** Overlapping forward returns must not straddle a split boundary.
- **Selection happens on validation, evaluation happens once on test.** The test slice never influences a choice.
- **Regime labels and z-scores use trailing windows only.** A label assigned at time `t` must be computable at time `t`.
- **Latency is enforced in the execution model.** A signal observed at `t` trades on the book at `t + 1` at the earliest, so a look-ahead strategy is structurally impossible to write.

## Success criteria

Two outcomes count as success, and only one of them involves finding an edge:

1. A short-horizon edge that survives costs, out-of-sample testing, and multiple-testing correction.
2. A demonstration that apparent alpha disappears under realistic costs and honest validation, with the specific point of failure identified.

The failure mode is neither of those: a large backtest number with no account of what would have made it go away.

## Known limitations

- **The circular-shift permutation null is mildly anti-conservative.** On the synthetic null control its distribution is roughly 40% narrower than the spread of rank IC across independent realisations, because shifting preserves each series' own autocorrelation but not the variation between realised paths. Single-sample p-values should be read as optimistic; the multi-seed control in `tests/test_synthetic_recovery.py` is the stronger check.
- **The IC shift test is exact; the gate-level randomisation null is not.** The IC table enumerates every admissible circular shift by cross-correlation, so it carries no sampling floor. The per-signal gates still sample 24 shifted runs, so their p-values floor at 0.04 and no signal can clear that gate more convincingly than that.
- **Multiple-testing correction is taken at the full family size**, 1,080, not at the size of the table displayed. Correcting over the 24 rows shown charges for a fraction of the searching actually done. When the shift test was sampled at 300 draws, its floor of 0.0033 exceeded the rank-24 threshold of 0.0011 and no rejection at the true family size was possible at all, whatever the data said.
- **Rank IC bootstraps resample pre-computed ranks.** Ranks are not recomputed inside each draw. This is a deliberate approximation that keeps several hundred bootstrap draws affordable on a full panel.
- **Annualised Sharpe assumes independence across events.** That is exactly the assumption the block bootstrap exists to doubt, so per-event Sharpe is the primary figure throughout.

## Amendments

**August 2026, post-audit.** Four defects found in the implementation, all of which the protocol above had asserted were handled and none of which were. Walk-forward folds and the nested-model split were taken by row position on an instrument-major panel, so each test block was a single ticker. The adverse-selection conditioner was ranked across the pooled panel, making it a proxy for instrument identity. Benjamini-Hochberg was applied at the size of the displayed table rather than the stated family of 1,080. All four are fixed, all four are documented in the paper rather than silently corrected, and the results that changed are recorded there.
