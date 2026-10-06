# Win rates: the five definitions of #226
_Source: plots/simulation/25_LEVIN_run1_ah_zero_pairings/per_round.parquet, plots/simulation/25_LEVIN_run3_sigmoid_pairings/per_round.parquet — 17 runs; head-to-head definitions pool both positions by matchup. Empty group-round = 0._

## 1. Payoff sum, head to head

Per round, the group's payoff summed over its members, averaged over the episode; the higher group wins the episode.

| matchup (a vs b) | episodes | a_win% | b_win% | a_mean | b_mean | a_empty% | b_empty% |
|---|---|---|---|---|---|---|---|
| ah vs ah | 100 | 54.0 | 46.0 | 94.15 | 83.81 | 9.5 | 11.7 |
| ah vs zero | 200 | 35.0 | 65.0 | 80.65 | 112.83 | 10.8 | 6.3 |
| zero vs zero | 100 | 43.0 | 57.0 | 96.57 | 104.08 | 8.8 | 6.8 |
| opt_pool vs zero | 100 | 50.0 | 50.0 | 93.69 | 104.24 | 8.3 | 7.5 |
| opt_pool vs ah | 100 | 60.0 | 40.0 | 112.25 | 83.17 | 6.3 | 12.7 |
| opt_pool vs rule_k1 | 100 | 61.0 | 39.0 | 111.28 | 81.59 | 5.2 | 10.8 |
| opt_pool vs rule_k2 | 100 | 63.0 | 37.0 | 114.73 | 81.74 | 6.2 | 16.0 |
| opt_pool vs rule_k4 | 100 | 46.0 | 54.0 | 96.29 | 103.98 | 10.2 | 9.3 |
| opt_pool vs rule_k8 | 100 | 45.0 | 55.0 | 83.7 | 111.48 | 10.2 | 6.8 |
| best_cap10_pool vs zero | 100 | 45.0 | 55.0 | 86.63 | 102.98 | 9.5 | 6.7 |
| best_cap10_pool vs ah | 100 | 68.0 | 32.0 | 115.47 | 72.27 | 4.3 | 13.7 |
| best_cap10_pool vs rule_k1 | 100 | 62.0 | 38.0 | 110.5 | 80.61 | 5.5 | 13.0 |
| best_cap10_pool vs rule_k2 | 100 | 57.0 | 43.0 | 103.81 | 94.02 | 8.3 | 11.8 |
| best_cap10_pool vs rule_k4 | 100 | 50.0 | 50.0 | 97.31 | 97.75 | 6.0 | 7.7 |
| best_cap10_pool vs rule_k8 | 100 | 55.0 | 45.0 | 99.05 | 97.23 | 7.5 | 9.0 |
| opt_pool vs best_cap10_pool | 100 | 51.0 | 49.0 | 99.29 | 99.18 | 7.7 | 8.7 |

## 2. Per-capita contribution, head to head

Per round, the mean contribution per member, averaged over the episode; the higher group wins the episode.

| matchup (a vs b) | episodes | a_win% | b_win% | a_mean | b_mean | a_empty% | b_empty% |
|---|---|---|---|---|---|---|---|
| ah vs ah | 100 | 57.0 | 43.0 | 8.4 | 8.14 | 9.5 | 11.7 |
| ah vs zero | 200 | 49.0 | 51.0 | 8.38 | 8.07 | 10.8 | 6.3 |
| zero vs zero | 100 | 46.0 | 54.0 | 7.18 | 7.38 | 8.8 | 6.8 |
| opt_pool vs zero | 100 | 51.0 | 49.0 | 9.14 | 8.23 | 8.3 | 7.5 |
| opt_pool vs ah | 100 | 53.0 | 47.0 | 10.68 | 9.75 | 6.3 | 12.7 |
| opt_pool vs rule_k1 | 100 | 56.0 | 44.0 | 11.26 | 10.57 | 5.2 | 10.8 |
| opt_pool vs rule_k2 | 100 | 62.0 | 38.0 | 10.3 | 9.27 | 6.2 | 16.0 |
| opt_pool vs rule_k4 | 100 | 51.0 | 49.0 | 9.98 | 9.71 | 10.2 | 9.3 |
| opt_pool vs rule_k8 | 100 | 48.0 | 52.0 | 8.66 | 8.56 | 10.2 | 6.8 |
| best_cap10_pool vs zero | 100 | 63.0 | 37.0 | 7.43 | 6.48 | 9.5 | 6.7 |
| best_cap10_pool vs ah | 100 | 63.0 | 37.0 | 9.2 | 8.0 | 4.3 | 13.7 |
| best_cap10_pool vs rule_k1 | 100 | 51.0 | 49.0 | 10.14 | 10.21 | 5.5 | 13.0 |
| best_cap10_pool vs rule_k2 | 100 | 49.0 | 51.0 | 9.36 | 9.73 | 8.3 | 11.8 |
| best_cap10_pool vs rule_k4 | 100 | 58.0 | 42.0 | 8.85 | 8.52 | 6.0 | 7.7 |
| best_cap10_pool vs rule_k8 | 100 | 58.0 | 42.0 | 8.69 | 7.73 | 7.5 | 9.0 |
| opt_pool vs best_cap10_pool | 100 | 56.0 | 44.0 | 10.03 | 9.76 | 7.7 | 8.7 |

## 3. Per-capita common good, head to head

Per round, the mean common good per member, averaged over the episode; the higher group wins the episode.

| matchup (a vs b) | episodes | a_win% | b_win% | a_mean | b_mean | a_empty% | b_empty% |
|---|---|---|---|---|---|---|---|
| ah vs ah | 100 | 52.0 | 48.0 | 11.87 | 11.55 | 9.5 | 11.7 |
| ah vs zero | 200 | 41.0 | 59.0 | 11.97 | 13.12 | 10.8 | 6.3 |
| zero vs zero | 100 | 45.0 | 55.0 | 11.7 | 12.01 | 8.8 | 6.8 |
| opt_pool vs zero | 100 | 46.0 | 54.0 | 13.5 | 13.37 | 8.3 | 7.5 |
| opt_pool vs ah | 100 | 56.0 | 44.0 | 16.18 | 14.33 | 6.3 | 12.7 |
| opt_pool vs rule_k1 | 100 | 59.0 | 41.0 | 17.13 | 14.85 | 5.2 | 10.8 |
| opt_pool vs rule_k2 | 100 | 59.0 | 41.0 | 15.62 | 13.81 | 6.2 | 16.0 |
| opt_pool vs rule_k4 | 100 | 48.0 | 52.0 | 14.85 | 15.24 | 10.2 | 9.3 |
| opt_pool vs rule_k8 | 100 | 43.0 | 57.0 | 12.6 | 13.7 | 10.2 | 6.8 |
| best_cap10_pool vs zero | 100 | 52.0 | 48.0 | 10.96 | 10.59 | 9.5 | 6.7 |
| best_cap10_pool vs ah | 100 | 64.0 | 36.0 | 14.0 | 11.45 | 4.3 | 13.7 |
| best_cap10_pool vs rule_k1 | 100 | 57.0 | 43.0 | 15.61 | 14.31 | 5.5 | 13.0 |
| best_cap10_pool vs rule_k2 | 100 | 51.0 | 49.0 | 14.37 | 14.75 | 8.3 | 11.8 |
| best_cap10_pool vs rule_k4 | 100 | 54.0 | 46.0 | 13.43 | 13.2 | 6.0 | 7.7 |
| best_cap10_pool vs rule_k8 | 100 | 55.0 | 45.0 | 13.25 | 12.32 | 7.5 | 9.0 |
| opt_pool vs best_cap10_pool | 100 | 54.0 | 46.0 | 15.03 | 15.07 | 7.7 | 8.7 |

## 4. Levin's criterion: common pool against `ah` (#219)

Per round, the group's common pool 1.6·Σc − Σp, averaged over the episode. Managers never meet: each plays `ah`, and **the one whose group does best against `ah` wins** (ranked below). Margin = its group minus `zero`'s group against `ah`, 95% bootstrap interval over episodes; members = mean group size.

> ⚠ not compared, never played `ah`: rule_k1, rule_k2, rule_k4, rule_k8

| manager | episodes | pool | members | vs zero [95%] |
|---|---|---|---|---|
| opt_pool | 100 | 78.1 | 4.51 | +17.1 [+6.2, +28.1] |
| best_cap10_pool | 100 | 70.2 | 4.76 | +9.2 [-1.1, +19.5] |
| zero (reference) | 200 | 60.9 | 4.45 | — |
| ah | 200 | 53.4 | 4.0 | -7.5 [-16.0, +0.8] |

## 5. Common pool, head to head

The same pool; the two groups of the same episode. Margin = a − b, paired by episode, 95% bootstrap interval.

| matchup (a vs b) | episodes | a_win% | a_pool | b_pool | margin a - b [95%] |
|---|---|---|---|---|---|
| ah vs ah | 100 | 58.0 | 57.1 | 49.7 | +7.4 [-6.1, +21.3] |
| ah vs zero | 200 | 39.0 | 50.1 | 60.9 | -10.9 [-19.5, -2.2] |
| zero vs zero | 100 | 44.0 | 48.7 | 54.5 | -5.8 [-17.9, +6.3] |
| opt_pool vs zero | 100 | 48.0 | 59.3 | 59.0 | +0.3 [-12.1, +13.0] |
| opt_pool vs ah | 100 | 63.0 | 78.1 | 57.8 | +20.2 [+5.3, +35.2] |
| opt_pool vs rule_k1 | 100 | 60.0 | 78.4 | 61.5 | +16.9 [+2.4, +31.4] |
| opt_pool vs rule_k2 | 100 | 64.0 | 76.7 | 56.9 | +19.8 [+4.2, +34.7] |
| opt_pool vs rule_k4 | 100 | 48.0 | 66.8 | 67.7 | -0.9 [-17.9, +15.8] |
| opt_pool vs rule_k8 | 100 | 41.0 | 51.3 | 65.7 | -14.4 [-27.9, -1.1] |
| best_cap10_pool vs zero | 100 | 48.0 | 46.5 | 46.4 | +0.0 [-10.8, +10.9] |
| best_cap10_pool vs ah | 100 | 66.0 | 70.2 | 42.9 | +27.3 [+14.8, +39.7] |
| best_cap10_pool vs rule_k1 | 100 | 62.0 | 70.7 | 61.2 | +9.6 [-6.8, +25.5] |
| best_cap10_pool vs rule_k2 | 100 | 56.0 | 65.4 | 63.9 | +1.5 [-12.6, +15.8] |
| best_cap10_pool vs rule_k4 | 100 | 53.0 | 58.5 | 57.5 | +1.1 [-11.9, +13.9] |
| best_cap10_pool vs rule_k8 | 100 | 55.0 | 59.8 | 52.1 | +7.7 [-5.1, +20.2] |
| opt_pool vs best_cap10_pool | 100 | 51.0 | 66.3 | 65.8 | +0.5 [-14.7, +15.5] |
