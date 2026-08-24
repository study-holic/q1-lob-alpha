# lobster_panel

Config: `configs/lobster_panel.yaml` (hash `6f988af81d1e`)  
Runtime: 114.8s

## Headline backtest

- signals: ofi, threshold 1.0, execution level 2 (cross_spread), latency 1 event(s)
- gross P&L 0.4217, net P&L -8.4296
- costs took 2099.1% of gross
- net Sharpe -0.1773 per event against gross 0.0317 per event
- annualised at sqrt of time that is -43438.2 net, which the block bootstrap should be read against rather than quoted alone
- 41,250 trades, exposure 23.6%, max drawdown -8.4296

## Walk forward

- 5.0 folds, 0.0 with positive test Sharpe
- mean test Sharpe -0.0181 against mean train Sharpe -0.0137 (per event)
- in-sample inflation 0.0044 Sharpe points per event

## Verdict

| signal               |   peak_abs_ic |   peak_horizon |   ic_mean_across_instruments |   best_evidence_horizon |   t_across_instruments |   instruments_same_sign | predictive   | survives_multiple_testing   | selected_out_of_sample   | positive_out_of_sample   |   gross_pnl |   net_pnl_at_baseline_cost | net_profitable_at_baseline_cost   |   breakeven_cost_multiple |   randomisation_pvalue_gross |   randomisation_pvalue_net |   randomisation_pvalue_floor | beats_shifted_null   | survives_all_gates   |
|:---------------------|--------------:|---------------:|-----------------------------:|------------------------:|-----------------------:|------------------------:|:-------------|:----------------------------|:-------------------------|:-------------------------|------------:|---------------------------:|:----------------------------------|--------------------------:|-----------------------------:|---------------------------:|-----------------------------:|:---------------------|:---------------------|
| ofi                  |    0.167879   |            100 |                   0.163665   |                      25 |               17.9932  |                     1   | True         | True                        | False                    | False                    |  0.421662   |                   -8.42961 | False                             |                 0.0476386 |                         0.04 |                       0.04 |                         0.04 | True                 | False                |
| queue_imbalance      |    0.292028   |            100 |                   0.149693   |                       5 |                7.3198  |                     1   | True         | True                        | False                    | False                    |  0.720087   |                  -17.9427  | False                             |                 0.0385841 |                         0.04 |                       0.04 |                         0.04 | True                 | False                |
| microprice_deviation |    0.306026   |            100 |                   0.104896   |                       1 |                7.5561  |                     1   | True         | True                        | True                     | False                    |  0.703811   |                  -16.6989  | False                             |                 0.0404426 |                         0.04 |                       0.04 |                         0.04 | True                 | False                |
| spread               |    0.00873291 |            100 |                   0.00516872 |                       1 |                1.06603 |                     0.6 | False        | False                       | True                     | False                    | -0.00673703 |                   -6.59672 | False                             |               nan         |                         0.08 |                       0.04 |                         0.04 | False                | False                |

Tables in `results/tables/lobster_panel`, figures in `results/figures/lobster_panel`.
