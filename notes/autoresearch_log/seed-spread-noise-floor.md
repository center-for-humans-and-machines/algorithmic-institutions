# The run-to-run noise floor of the 22-row evaluation

Branch `auto/seed-spread-noise-floor`, based on `origin/auto/punisher-ceiling-fix` at `e230629` (PR #192, the maintainer's current model). A measurement, not an experiment: no model is proposed, no row is declared a target, and no §2 gate applies. The deliverable is an error bar on every row of the evaluation, so that past and future single-seed verdicts can be read against it.

## 1. The question

Every verdict in this project rests on one training run and one simulation, and the protocol has no notion of run-to-run variability. The three most recent experiments were decided on margins that may sit inside it: PR #194's target row finished 8.6% of one noise ceiling short of a band; PR #193's protected-row firing was 1.14 pooled standard errors; PR #194's clearest erosion was 1.96. PR #194 also could not separate whether its new feature shifted the simulated contribution level by a whole point, from 9.32 to 10.36 against the human 9.457, or whether that particular retrain did — its own caveat section says so and asks for two more seeds. Without a noise floor none of these can be read.

The question is therefore not "is model X better" but: **how far does the whole 22-row evaluation move when the only thing that changes is the random seed of a training that everybody already accepts?**

## 2. Setup

**No training.** PR #188 (`origin/auto/copula-seed-ensemble`) already trained five copies of the stimulus-skip contributor with seeds 1–5, the shipped config byte-for-byte, on the full training data. Its log records their cross-validated log losses as 2.0226 / 2.0293 / 2.0221 / 2.0201 / 2.0253 against the shipped artifact's 2.0206, and that the shipped artifact sits as far from the ensemble mean as any member does. The committed metrics parquets on this branch reproduce those numbers (2.0216 / 2.0284 / 2.0221 / 2.0201 / 2.0253). So the shipped contributor is a sixth draw from the same distribution, and the six arms are exchangeable by construction.

**The six arms.** The frontier stack of PR #192, `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling` — the stimulus-skip contributor with its copula, the joint-exodus graph-network switch model, and the ceiling-fixed `lin_multinomial` punisher with its severity copula — with **only** the contribution artifact swapped. The five arm configs differ from the source config in exactly three lines: `contribution_model`, `output_dir`, `figure_name`. The sixth arm is the shipped run itself, the one PRs #193 and #194 used as their baseline, so the spread is anchored on the number those experiments compared against.

**The copula is carried, not recalibrated.** The five members are bare; the frontier contributor is copula-stamped. Each member therefore gets the shipped calibration's `rho = 0.03949863621805423`, `phi_final = 1.0`, `copula_switch_every = 1` copied bit for bit (`carry_contribution_copula_params.py`, taken unchanged from `origin/auto/contributor-ceiling-indicator`, then `make_contribution_copula_artifact.py`). Parameters are frozen per model family when only the marginal changes; recalibrating would have made the arms differ in two things instead of one. Every stamp is verified: all 14 tensors bit-identical to the bare base, and the teacher-forced probabilities on all 7,457 train-split rows unchanged.

**Load-and-differ check** (`scripts/data_analysis/seed_spread_verify_members.py`, Raven login node). All six arms load through `GraphNetwork.load`, carry `y_name = contribution`, have the same 7,061 parameters and the same three copula fields. Pairwise max |delta| over the parameter vector 2.06–3.24, relative L2 ~1.4 — independent draws, not perturbations. The shipped artifact's distance to the members (2.45–3.08) sits inside the members' distance to each other (2.06–3.24), which is PR #188's "sixth draw" finding reproduced on the weights.

**Simulation.** The 23-family protocol untouched: seed 42, 100 episodes, 24 rounds, 2 groups x 8 agents, `save_per_round: true`. The simulation seed and episode count are identical across all six arms, so what varies is the trained model and not the draw. Raven, isolated dir `AI_REMOTE_DIR=~/repros/ai-runs/seed-spread`, one A100 each.

**Evaluation.** All 22 rows, locally, with `PYTHONPATH=<worktree>/src` — the shared venv's editable install resolves `aimanager` to the main checkout, which lacks the RCE row and silently scores 21.

**Analysis.** `scripts/data_analysis/seed_spread_noise_floor.py` writes `plots/data_analysis/evaluation/seed_spread_noise_floor/`.

## 3. Results

Everything in this section is measured, from the six arms' own `evaluation/scores.csv` and `per_round.parquet`. Inference is section 5. Full tables: `plots/data_analysis/evaluation/seed_spread_noise_floor/` (`per_row.csv/.md`, `ceiling_stability.csv`, `aggregates.csv`, `arm_ranks.csv`, `levels.csv`, `levels_spread.csv`, `verdicts.csv`, `rce_band_se.csv`, `rce_band_n.csv`, `per_row_spread.jpg`).

### Step 1: the six arms are sound and genuinely different (measured)

All six `per_round.parquet` have the same shape, 19,200 rows = 100 episodes x 24 rounds x 8 agents, the same ten columns, one `run` each (`ah group_switching managed by lin_multinomial_copula_self`). All six `evaluation/scores.csv` carry all 22 rows for that run, RCE included — the evaluations were run locally with `PYTHONPATH=<worktree>/src`, and the 21-row failure mode did not occur. The noise-ceiling denominators are identical across the six to floating point (asserted in `load_scores`; they are a human-vs-human quantity and must not depend on which contributor ran, or the six scores are not comparable).

The five members are not duplicates of each other or of the shipped run. Pairwise over the 19,200 contribution entries, 11.3% to 20.2% coincide (chance-level for this marginal) and the mean absolute difference is 5.26 to 6.65 contribution units; all fifteen pairs differ, and all six frames have distinct content hashes. This is on top of the weight-level check already in section 2 (pairwise max |delta| 2.06-3.24 over the parameter vector).

The shipped arm is the *same* run PRs #193 and #194 baselined against, not a re-run of it: its scores reproduce PR #194's "before" column digit for digit — CA 0.856264, CB 0.794091, CC 0.895775, CD 0.794941, CG 1.758781, PD 0.759813, RCA 1.652555, RCB 1.659051, RCC 1.296895, RCD 1.251543, RCE 0.882320, mean 1.0331, rows <= 1 14/22, mean contribution 9.3235. The spread below is therefore anchored on exactly the number the recent experiments compared against.

### Step 2: the per-row spread over the six arms (measured)

`sd_to_nearest_bound` is the distance from the six-arm mean to the nearest scoring-band boundary (1 / 2 / 5) in units of the seed sd; `gateable` is that distance being at least 1.

| row | seed_1 | seed_2 | seed_3 | seed_4 | seed_5 | shipped | mean | sd | range | bands | sd_to_nearest_bound | gateable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CA | 1.1233 | 1.4033 | 1.0349 | 1.0551 | 0.9466 | 0.8563 | 1.0699 | 0.1878 | 0.5471 | <= 1/1-2 | 0.3723 | **no** |
| CB | 1.1019 | 1.4431 | 1.0112 | 1.0693 | 0.9599 | 0.7941 | 1.0632 | 0.2153 | 0.6490 | <= 1/1-2 | 0.2938 | **no** |
| CC | 1.0307 | 1.2376 | 0.9699 | 0.9541 | 0.8661 | 0.8958 | 0.9924 | 0.1333 | 0.3715 | <= 1/1-2 | 0.0572 | **no** |
| CD | 1.0919 | 1.3541 | 1.0031 | 1.0365 | 0.9226 | 0.7949 | 1.0339 | 0.1878 | 0.5591 | <= 1/1-2 | 0.1803 | **no** |
| CE | 1.1776 | 1.1422 | 1.0935 | 1.2192 | 1.1560 | 1.0579 | 1.1411 | 0.0581 | 0.1614 | 1-2 | 2.4294 | yes |
| CF | 1.0780 | 1.1262 | 0.9062 | 1.1748 | 0.9283 | 0.8169 | 1.0051 | 0.1413 | 0.3578 | <= 1/1-2 | 0.0358 | **no** |
| CG | 1.0956 | 1.0050 | 0.9894 | 0.9920 | 1.2809 | 1.7588 | 1.1870 | 0.3014 | 0.7694 | <= 1/1-2 | 0.6202 | **no** |
| SA | 1.0039 | 0.7459 | 0.7761 | 1.1533 | 0.9063 | 0.7687 | 0.8924 | 0.1618 | 0.4073 | <= 1/1-2 | 0.6650 | **no** |
| SB | 1.0687 | 1.0335 | 0.9360 | 0.9784 | 1.0035 | 1.0105 | 1.0051 | 0.0456 | 0.1328 | <= 1/1-2 | 0.1120 | **no** |
| SC | 1.1688 | 1.1950 | 1.4446 | 1.4600 | 1.3836 | 1.4632 | 1.3525 | 0.1355 | 0.2945 | 1-2 | 2.6015 | yes |
| PA | 0.6731 | 0.7556 | 0.6590 | 0.7210 | 0.6739 | 0.6526 | 0.6892 | 0.0404 | 0.1030 | <= 1 | 7.6868 | yes |
| PB | 0.9138 | 0.9319 | 0.9005 | 0.9594 | 0.9366 | 0.9558 | 0.9330 | 0.0231 | 0.0589 | <= 1 | 2.9047 | yes |
| PC | 0.9197 | 0.8956 | 0.9511 | 0.9958 | 0.9086 | 0.9349 | 0.9343 | 0.0359 | 0.1002 | <= 1 | 1.8327 | yes |
| PD | 0.7793 | 0.7414 | 0.7055 | 0.8551 | 0.8470 | 0.7598 | 0.7813 | 0.0593 | 0.1496 | <= 1 | 3.6895 | yes |
| RCA | 1.4618 | 1.7491 | 1.4137 | 1.4589 | 1.4136 | 1.6526 | 1.5249 | 0.1412 | 0.3356 | 1-2 | 3.3646 | yes |
| RCB | 1.3904 | 1.3118 | 1.4241 | 1.6351 | 1.5693 | 1.6591 | 1.4983 | 0.1425 | 0.3472 | 1-2 | 3.4971 | yes |
| RCC | 1.7554 | 1.4909 | 1.6309 | 1.5777 | 1.6888 | 1.2969 | 1.5734 | 0.1631 | 0.4585 | 1-2 | 2.6151 | yes |
| RCD | 1.7157 | 1.9905 | 1.6215 | 1.9688 | 1.7271 | 1.2515 | 1.7125 | 0.2698 | 0.7390 | 1-2 | 1.0656 | yes |
| RCE | 1.0520 | 1.0956 | 1.0887 | 1.2110 | 1.0804 | 0.8823 | 1.0683 | 0.1063 | 0.3287 | <= 1/1-2 | 0.6427 | **no** |
| RSA | 1.1786 | 1.1407 | 0.9265 | 1.0258 | 1.3437 | 0.9653 | 1.0968 | 0.1555 | 0.4172 | <= 1/1-2 | 0.6223 | **no** |
| RPA | 0.6778 | 0.6624 | 0.6438 | 0.6444 | 0.6875 | 0.6620 | 0.6630 | 0.0175 | 0.0437 | <= 1 | 19.2437 | yes |
| RPB | 0.7655 | 0.8434 | 0.8004 | 0.8167 | 0.8189 | 0.8380 | 0.8138 | 0.0283 | 0.0780 | <= 1 | 6.5703 | yes |

The typical row moves by 0.138 (median sd; range of sds 0.018 to 0.301) and spans 0.341 end to end (median range; 0.044 to 0.769) on the seed alone. Picture: `per_row_spread.jpg`.

### Step 3: the aggregates (measured)

`within_run_se` is the median within-run sampling SE of that band slope across the six arms — the error bar PRs #193 and #194 already quoted, next to the seed spread.

| quantity | seed_1 | seed_2 | seed_3 | seed_4 | seed_5 | shipped | mean | sd | min | max | range | within_run_se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_22 | 1.1011 | 1.1498 | 1.0423 | 1.1347 | 1.0931 | 1.0331 | 1.0923 | 0.0473 | 1.0331 | 1.1498 | 0.1167 | |
| rows_le1 | 6 | 7 | 12 | 9 | 12 | 14 | 10.0000 | 3.1623 | 6 | 14 | 8 | |
| rce_slope_0-4 | 0.0432 | 0.0441 | 0.0615 | 0.0454 | 0.0737 | 0.0868 | 0.0591 | 0.0182 | 0.0432 | 0.0868 | 0.0436 | 0.0143 |
| rce_slope_5-9 | 0.0490 | 0.0592 | 0.0187 | 0.0057 | 0.0078 | 0.0379 | 0.0297 | 0.0223 | 0.0057 | 0.0592 | 0.0535 | 0.0145 |
| rce_slope_10-14 | 0.0015 | 0.0371 | -0.0095 | 0.0043 | -0.0163 | -0.0427 | -0.0043 | 0.0263 | -0.0427 | 0.0371 | 0.0798 | 0.0197 |
| rce_slope_15-19 | -0.1006 | -0.0839 | 0.0111 | -0.0650 | 0.0012 | -0.1298 | -0.0612 | 0.0564 | -0.1298 | 0.0111 | 0.1409 | 0.0417 |

The gate-2 margin is 10% of the baseline mean, 0.1033 here. The seed sd of the 22-row mean is 0.0473 and its six-arm range is 0.1167 — the whole gate-2 allowance is 2.2 seed sd wide, and the observed range of same-model retrains already exceeds it.

Two of the four RCE band slopes change sign across the six arms with nothing but the seed changed: 10-14 runs +0.037 to -0.043, and 15-19 runs +0.011 to -0.130. The protected-row rule forbids flipping a band slope away from the human sign (human: + + - -), so on a single run that clause is decided by the draw on both negative bands. The seed sd exceeds the within-run sampling SE on every band (1.27x, 1.53x, 1.34x, 1.35x): retraining moves a band slope more than resampling the same run does.

### Step 4: rows at or under the noise ceiling (measured)

| status | rows |
|---|---|
| always <= 1 in all six arms | PA, PB, PC, PD, RPA, RPB (6) |
| never <= 1 in any arm | CE, SC, RCA, RCB, RCC, RCD (6) |
| flips on the seed alone | CA, CB, CC, CD, CF, CG, SA, SB, RCE, RSA (10) |

Per-row counts of arms at or under the ceiling: CA 2, CB 2, CC 4, CD 2, CE 0, CF 3, CG 2, SA 4, SB 2, SC 0, PA 6, PB 6, PC 6, PD 6, RCA 0, RCB 0, RCC 0, RCD 0, RCE 1, RSA 2, RPA 6, RPB 6.

The reported count is the column sum of those flips: 6 / 7 / 12 / 9 / 12 / 14 over the six arms, mean 10.0, **sd 3.16, range 8**. Ten of the 22 rows are on the wrong side of the ceiling in some arms and the right side in others, with no change to the model family, the data, the protocol or the simulation seed.

### Step 5: rows that cannot be gated on a single run (the actionable output)

A scoring band boundary lies inside one seed sd of the six-arm mean on **ten of the 22 rows**: **CA, CB, CC, CD, CF, CG, SA, SB, RCE, RSA**. On every one of them the six arms actually land in two different bands (`<= 1/1-2` throughout — no row reaches the 2 or 5 boundary). Distance to the nearest boundary in seed sd: CF 0.04, CC 0.06, SB 0.11, CD 0.18, CB 0.29, CA 0.37, CG 0.62, RSA 0.62, RCE 0.64, SA 0.67.

The twelve that can be gated, with their margin in seed sd: RPA 19.2, PA 7.7, RPB 6.6, PD 3.7, RCB 3.5, RCA 3.4, PB 2.9, RCC 2.6, SC 2.6, CE 2.4, PC 1.8, **RCD 1.07** (marginal).

**The protected row RCE is in the ungateable set.** Its band is `<= 1` in one arm of six (the shipped one) and `1-2` in the other five, the boundary sits 0.64 seed sd from the mean, and two of its four band slopes change sign across the arms. Every clause of the protected-row rule — band drop, sign flip, halved magnitude — is decided by the training draw on a single run.

**The rows <= 1 count cannot be gated either**, and it is worse than any individual row: sd 3.16 on a 22-row scale, range 6 to 14. §3 ranks stacks by this count (descending, ties broken by the lower mean) to pick the evaluation stack, and §10 reports it in every results table.

### Step 6: the contribution level — PR #194's open question (measured)

| arm | mean c | share c = 0 | share c = 20 | mean p |
|---|---|---|---|---|
| human | **9.4572** | **9.36%** | 13.44% | 1.7913 |
| shipped (the baseline PR #194 compared against) | 9.3235 | 8.21% | 12.53% | 1.8014 |
| seed_1 | 10.3270 | 5.79% | 16.20% | 1.6711 |
| seed_2 | 10.7809 | 5.35% | 17.15% | 1.6427 |
| seed_3 | 10.1620 | 7.06% | 14.89% | 1.7659 |
| seed_4 | 10.1711 | 5.55% | 17.38% | 1.6516 |
| seed_5 | 10.0399 | 8.52% | 16.16% | 1.6848 |
| six-arm mean / sd / range | 10.1341 / 0.4740 / 1.4573 | 6.75% / 1.39pp / 3.17pp | 15.72% / 1.79pp / 4.84pp | 1.7029 / 0.0652 / 0.1588 |
| **PR #194's candidate** | **10.3550** | **5.4%** | 12.3% | 1.6240 |

Six contributors of the same architecture, trained on the same data with the same config, span **9.32 to 10.78** in simulated mean contribution — a range of 1.46, larger than the 1.03 PR #194 moved. PR #194's candidate at 10.355 sits **0.20 sd from the five members' mean** (10.296 +- 0.289); the shipped baseline at 9.324 sits **3.36 sd below it**. On the share giving nothing the same holds: the candidate's 5.4% is bracketed by seed_2's 5.35% and seed_4's 5.55%, and reproduced to within 0.4 pp by three of the five members.

A plain reseed of the accepted contributor — no feature, no code change, nothing but seed 1 instead of 38381 — lands at mean contribution **10.327** and share-zero **5.79%**, against PR #194's candidate's **10.355** and **5.40%**.

### Step 7: the four recent verdicts against the floor (measured)

`in_seed_sd` is |after - before| divided by the seed sd of that quantity; `after_in_arm_range` is whether the quoted "after" value falls inside the six arms' own min-max. Movements on #190, #192 and #193 come from stacks whose punisher or switch slot also differs, so the seed sd is the contributor-retrain floor applied to them, not a full error bar.

| pr | movement | before | after | delta | seed_sd | in_seed_sd | inside floor | after in arm range |
|---|---|---|---|---|---|---|---|---|
| #190 | SC (declared target, no band) | 1.4270 | 1.3290 | -0.0980 | 0.1355 | 0.72 | **inside** | yes |
| #190 | RCD (declared target, wrong way) | 1.3090 | 1.6890 | +0.3800 | 0.2698 | 1.41 | outside | yes |
| #190 | RSA (largest regression) | 1.0700 | 1.6440 | +0.5740 | 0.1555 | 3.69 | outside | no |
| #190 | SB (band upgrade, not declared) | 1.0060 | 0.8980 | -0.1080 | 0.0456 | 2.37 | outside | no |
| #190 | RCE 10-14 slope (protected row fired) | -0.0580 | -0.0200 | +0.0380 | 0.0263 | 1.44 | outside | yes |
| #192 | RCC (declared target, no band) | 1.5298 | 1.2969 | -0.2329 | 0.1631 | 1.43 | outside | yes |
| #192 | 22-row mean | 1.0357 | 1.0331 | -0.0026 | 0.0473 | 0.05 | **inside** | yes |
| #192 | rows <= 1 | 13 | 14 | +1 | 3.1623 | 0.32 | **inside** | yes |
| #192 | RCB (worsened, flagged) | 1.5454 | 1.6591 | +0.1137 | 0.1425 | 0.80 | **inside** | no |
| #192 | CG (worsened, flagged) | 1.5535 | 1.7588 | +0.2053 | 0.3014 | 0.68 | **inside** | no |
| #193 | RCC (declared target, wrong way) | 1.2969 | 1.4237 | +0.1268 | 0.1631 | 0.78 | **inside** | yes |
| #193 | RPA (declared target, wrong way) | 0.6620 | 0.6932 | +0.0312 | 0.0175 | 1.78 | outside | no |
| #193 | RCE 10-14 slope (protected row fired) | -0.0430 | -0.0070 | +0.0360 | 0.0263 | 1.37 | outside | yes |
| #193 | 22-row mean (best recorded) | 1.0331 | 0.9970 | -0.0361 | 0.0473 | 0.76 | **inside** | no |
| #193 | CG (band upgrade, not declared) | 1.7588 | 0.9655 | -0.7933 | 0.3014 | 2.63 | outside | no |
| #194 | RCC (declared target, 8.6% short) | 1.2969 | 1.0857 | -0.2112 | 0.1631 | 1.29 | outside | no |
| #194 | RCE (protected row, band drop) | 0.8823 | 1.1058 | +0.2235 | 0.1063 | 2.10 | outside | yes |
| #194 | RCE 0-4 slope (clearest erosion) | +0.0870 | +0.0480 | -0.0390 | 0.0182 | 2.15 | outside | yes |
| #194 | RCE 5-9 slope | +0.0380 | +0.0140 | -0.0240 | 0.0223 | 1.08 | outside | yes |
| #194 | RCE 15-19 slope | -0.1300 | -0.0370 | +0.0930 | 0.0564 | 1.65 | outside | yes |
| #194 | CA (collateral, left band <= 1) | 0.8560 | 1.2050 | +0.3490 | 0.1878 | 1.86 | outside | yes |
| #194 | CB (collateral, left band <= 1) | 0.7940 | 1.1480 | +0.3540 | 0.2153 | 1.64 | outside | yes |
| #194 | CC (collateral, left band <= 1) | 0.8960 | 1.0750 | +0.1790 | 0.1333 | 1.34 | outside | yes |
| #194 | CD (collateral, left band <= 1) | 0.7950 | 1.1740 | +0.3790 | 0.1878 | 2.02 | outside | yes |
| #194 | PD (collateral, left band <= 1) | 0.7598 | 1.0933 | +0.3335 | 0.0593 | 5.63 | outside | **no** |
| #194 | 22-row mean | 1.0331 | 1.0983 | +0.0652 | 0.0473 | 1.38 | outside | yes |
| #194 | mean contribution level | 9.3240 | 10.3550 | +1.0310 | 0.4740 | 2.17 | outside | yes |
| #194 | share giving nothing | 8.20% | 5.40% | -2.80pp | 1.39pp | 2.01 | outside | yes |

Eight of the 28 movements are inside one seed sd outright. Of the twenty outside, sixteen still land on an "after" value that some same-architecture retrain of the baseline reaches on its own.

### Step 8: where each arm sits among the six (measured)

| arm | rows where it is best of six | rows where it is worst | 22-row mean (rank) | rows <= 1 (rank) | \|mean c - human\| |
|---|---|---|---|---|---|
| seed_1 | 2 | 2 | 1.1011 (4) | 6 (6) | 0.870 |
| seed_2 | 3 | 8 | 1.1498 (6) | 7 (5) | 1.324 |
| seed_3 | 6 | 0 | 1.0423 (2) | 12 (2=) | 0.705 |
| seed_4 | 0 | 7 | 1.1347 (5) | 9 (4) | 0.714 |
| seed_5 | 2 | 2 | 1.0931 (3) | 12 (2=) | 0.583 |
| **shipped** | **9** | 3 | **1.0331 (1)** | **14 (1)** | **0.134** |

The shipped contributor is first of six on every headline aggregate: lowest 22-row mean, most rows at the ceiling, best on more individual rows than any member, and five times closer to the human contribution level than the nearest member. It holds the six-arm **minimum** on nine rows including **RCC (1.2969 against the members' 1.4909-1.7554) and RCE (0.8823, the only arm of six at or under the ceiling on that row)**.

## 4. Verdict

**The noise floor of the 22-row evaluation, with the simulation draw held fixed and only the contributor's training seed changed: a typical row moves by sd 0.138 and spans 0.34 end to end; the 22-row mean moves by sd 0.047 with a six-arm range of 0.117; the rows <= 1 count moves by sd 3.16 with a range of 8 (6 to 14); and the simulated contribution level moves by sd 0.47 with a range of 1.46.**

**Ten of 22 rows cannot be gated on a single run** — CA, CB, CC, CD, CF, CG, SA, SB, RCE, RSA — because a band boundary lies inside one seed sd of their mean and the six arms genuinely straddle it. **The protected row RCE is one of them.** Twelve rows can: RPA, PA, RPB, PD, RCB, RCA, PB, RCC, SC, CE, PC, and marginally RCD.

**PR #194's level question is answered: that was the retrain, not the feature.** Reseeding the accepted contributor and changing nothing else lands at mean contribution 10.327 and share-zero 5.79%, against PR #194's candidate's 10.355 and 5.40%; the candidate sits 0.20 sd from the five members' mean while the shipped baseline it was compared against sits 3.36 sd below it. Four of the five collateral rows PR #194 was faulted for (CA, CB, CC, CD) reach their "after" values in some arm of the reseeded ensemble. **PD is the exception** and the one piece of that collateral the floor does not explain: 0.7598 -> 1.0933 is 5.63 seed sd, and the after value lies outside the six arms' whole range (0.7055-0.8551).

**No verdict is reversed here and none should be.** The gate PR #194 failed was gate 1 — RCC did not clear a band — and RCC is one of the twelve gateable rows; the target genuinely did not upgrade. What the floor removes is the *reason* the collateral was read as damage caused by the feature.

## 5. Notes

Measured facts are in section 3. Everything below is inference from them.

1. **The baseline the last three experiments were judged against is the best of six same-architecture draws, and that is not a coincidence I can rule out.** PR #188 established that the shipped artifact's cross-validated log loss (2.0206) is as typical as the members' (2.0201-2.0284) — on the training objective it is an ordinary draw. On the simulation evaluation it is first of six on every aggregate and the minimum on nine rows. The obvious mechanism is selection: the shipped contributor became the frontier by scoring well on this evaluation, while the ensemble members never faced it. If that is right, the bias is structural rather than accidental — every accepted model is the tail of its own draw, and every candidate afterwards is measured against that tail. This would be worth one cheap check: whether the *next* accepted model also lands at the extreme of its own reseed ensemble.

2. **A candidate is therefore fighting a handicap on exactly the two rows the recent verdicts turned on.** RCC's baseline 1.2969 is the six-arm minimum and 1.7 sd below the arm mean; RCE's baseline 0.8823 is the six-arm minimum and the only arm of six at or under the ceiling. A candidate must beat a favourable draw on the target row and avoid dropping a band on a protected row whose baseline band only one arm of six achieves. PR #194's RCE "band drop" is largely this: its after value 1.1058 sits inside the members' ordinary range of 1.0520-1.2110, and five of six reseeds of the *accepted* model would also have been scored as a band drop against the shipped baseline.

3. **The rows <= 1 count should not be quoted as a property of a model.** Its seed sd is 3.16 on a 22-row scale, which is larger than any difference between stacks that the score matrix currently distinguishes. §2 already says it is context and not a criterion, and that is right; but §3 ranks stacks by it to choose the evaluation stack, so a quantity that moves by +-3 rows on the seed alone is picking which stack a candidate is judged in. Six rows are always at the ceiling and six never are; the count is a report on the ten rows in between, all of which sit on the boundary.

4. **The RCE band-slope clauses are below the resolution of a single run.** The seed sd exceeds the within-run sampling SE on all four bands, and two of the four slopes change sign across six retrains of the same model. PR #192's log already suspected this from the within-run SEs alone ("on one seed this is not distinguishable from noise") and asked for an absolute floor. The measurement says the problem is worse than the within-run SEs suggested, because retraining moves the slopes 1.3-1.5x more than resampling does. The sign clause in particular cannot survive: on the 10-14 and 15-19 bands the sign of an unchanged model is a coin flip.

5. **The 10% gate-2 margin is about two seed sd wide.** The margin here is 0.1033 and the seed sd of the 22-row mean is 0.0473, so the allowance corresponds to roughly a 2-sigma tolerance on a quantity nobody measured the sigma of. That is, by luck, about the right order: gate 2 is the one criterion in §2 whose tolerance is already commensurate with the floor. Gate 1 and the protected row have no tolerance at all, which is why they are the ones this measurement embarrasses.

6. **What the floor does not cover.** This is the *training-seed* component with the simulation seed fixed at 42, one contributor slot, one stack, six draws. It does not measure the simulation-draw component (a different `seed:` in the sim config), the punisher or switch slots' retrain spread, or any interaction between them. Six draws give an sd with roughly 30% relative uncertainty, so the numbers here are the right order of magnitude and not three-decimal constants. The direction of every likely omission is the same: the total run-to-run variability is larger than what is measured here, not smaller.

7. **The one thing on PR #194 that the floor does not excuse is PD.** 0.7598 -> 1.0933 is 5.63 seed sd on the tightest-but-one row in the suite, and the after value is outside the arms' entire range. If a successor revisits the contributor ceiling indicator, PD is the collateral to explain; CA/CB/CC/CD and the level shift are not evidence about the feature at all.

## 6. For a successor

1. **Recommendation to the maintainer: require a declared target to clear the floor, and report a seed band on every row.** Concretely, three changes to §2/§3, in descending order of value. (a) *Gate 1 gains a magnitude condition*: a band upgrade on a target row counts only if the row also moved by more than one seed sd of that row (the per-row sd column of `per_row.csv`), which costs nothing to check and rules out upgrades bought by a lucky draw on a row whose boundary sits 0.04 sd away. (b) *Every results table carries the row's seed sd next to the score*, and rows in the ungateable ten are marked, so a reader sees immediately whether a movement is legible. (c) *The protected-row rule gets the same treatment*: RCE's band-drop clause fires only on a drop larger than its seed sd (0.106), the sign clause is retired on the 10-14 and 15-19 bands where an unchanged model flips sign, and the magnitude clause keeps PR #192's proposed absolute floor.

2. **Why not "a second seed for any candidate landing inside the floor".** It is the statistically cleanest option and it should stay on the table, but it doubles the cost of exactly the experiments that are already marginal, it needs a rule against best-of-two reporting that §10 currently supplies only as an honour system ("no re-running for a better draw"), and — the decisive objection — it leaves the *baseline* single-seeded. The asymmetry documented in note 2 is the larger error: two seeds on the candidate against one favourable seed on the baseline would make the comparison worse, not better. If the maintainer does want retraining, the right version is a second seed on **both** sides, declared in advance, with both runs reported.

3. **The cheapest structural fix, if the maintainer will spend once: re-baseline the frontier stack on the ensemble mean rather than the shipped draw.** The six arms already exist and are committed here. Using the six-arm mean and sd per row as the baseline removes the selection bias of note 1 in one step and gives every future candidate a per-row error bar for free. This changes the recorded frontier numbers (mean 1.0923 rather than 1.0331, rows <= 1 10 rather than 14, RCC 1.5734 rather than 1.2969, RCE 1.0683 rather than 0.8823) and is therefore a maintainer decision, not an agent's: it would move three open PRs' baselines and it makes the frontier look worse while making it honest.

4. **Do not re-run any of this.** Six simulations, their evaluations and the analysis are committed. `plots/data_analysis/evaluation/seed_spread_noise_floor/per_row.csv` carries the per-row seed sd any successor needs; the script regenerates every table from the committed artifacts in about a minute with no cluster.

5. **What a successor to PR #194 should do differently.** Declare the level rows (CA/CB/CC/CD and the mean contribution) as watch items with their seed sd attached, and treat only PD and the RCC target as evidence about the feature. The interaction variant `p_{t-1} x I(c_{t-1} = 20)` that PR #194 recommends is still the right next step; what changes is that it no longer has to answer for a level shift that a reseed reproduces.

6. **Housekeeping.** Nothing was trained, no copula was recalibrated, and no cluster job was run by this session — the five simulations were fetched by the previous session before its login expired. The remote dir `~/repros/ai-runs/seed-spread` can be deleted when this PR closes.
