# Metrics for algoinst model evaluation

## From simulation

We simulate a large number of episodes with the trained models and compare the statistics of the simulation against the human experiment.

Rollouts involve three models (manager, contribution, switching), so credit cannot be cleanly assigned. Contribution statistics say the most about the contribution model, switching statistics about the switching model, and punishment statistics about the manager. The punishment rows test fidelity only where the simulated manager is meant to mimic humans; for an RL manager they characterise the policy rather than score it.

### Distributions

Marginalised distributions of the observed quantities — whether the model produces the right states. Drop (game, round) cells in which a group is empty.

#### Contribution (C)

* **CA** — over rounds → participant mean contributions (between-participant variability)
* **CB** — over participants → round mean contributions (temporal dynamics)
* **CC** — over participants and rounds, by group → group mean contributions (between-group variability)
* **CD** — none → raw contributions (natural contribution distribution)
* **CE** — over participants within group, then difference between the two groups of a game → signed group contribution differences over (game, round)
* **CF** — over participants, by round and boundary → share of contributions at 0 and at 20 per round (polarisation)

#### Switching (S)

* **SA** — none → switch rate over all switching opportunities
* **SB** — over participants, by opportunity → switch rate per switching opportunity (temporal dynamics)
* **SC** — over rounds → size of the larger group over (game, round) (segregation)

#### Punishment (P)

* **PA** — none → raw punishments (natural punishment distribution)
* **PB** — over participants → mean punishment per round
* **PC** — over participants, by round → share of punishments equal to zero per round (the extensive margin: whether to punish, as against how much)

### Responses (R)

A response conditioned on the stimulus it reacts to — whether the model has the right mechanisms. Contribution change means `Δc = c_{t+1} − c_t`, with the stimulus taken at round `t`.

* **RCA** — contribution change, by round type: no switch allowed / switched / chose to stay / stayed but group composition changed
* **RCB** — contribution change, by punishment rate, over punished non-full contributors (reaction to punishment)
* **RCC** — contribution change of full contributors, punished minus unpunished (reaction at the ceiling, where RCB's rate is undefined)
* **RCD** — switching pull: contribution change against the gap to the receiving group. For each switch event, let round *n* be the last round in the old group and *n+1* the first in the new; with `Ĉ` the receiving group's mean contribution in round *n*, regress `C_{n+1} − C_n ~ Ĉ − C_n`, and the slope is the pull coefficient
* **RSA** — switching, by punishment received in the previous round, over punished contributors only (who switches, not how many)
* **RPA** — punishment, by the contributor's contribution (the manager's policy)
* **RPB** — punishment, by group size, over rounds 4 onward

### Metrics

|     | 1. abs. Δ statistic | 2. abs. Δ std | 3. EMD |
|-----|---------------------|---------------|--------|
| CA  | | ✔ | ★ |
| CB  | ★ conditional on round | | |
| CC  | | ✔ | ★ |
| CD  | | | ★ |
| CE  | | ✔ | ★ |
| CF  | ★ conditional on round and boundary | | = column 1 |
| SA  | ★ switch rate | | |
| SB  | ★ conditional on opportunity | | |
| SC  | | | ★ |
| PA  | | | ★ |
| PB  | ★ conditional on round | | |
| PC  | ★ conditional on round | | = column 1 |
| RCA | | | ★ conditional on round type |
| RCB | ★ conditional on punishment rate | | |
| RCC | ★ punished-minus-unpunished contrast | | |
| RCD | ★ pull coefficient | | |
| RSA | ★ conditional on punishment | | = column 1 |
| RPA | | | ★ conditional on contribution |
| RPB | | | ★ conditional on group size |

★ marks the canonical score for the row. The three ✔ cells are the only diagnostic retained, reported **signed and in raw units** rather than as normalised scores.

Strata, for the cells marked *conditional on …*:

* **CB, PB, PC** — the 24 rounds, one stratum each; weights uniform
* **CF** — the 48 cells of {contribution = 0, contribution = 20} × 24 rounds; weights uniform
* **SB** — the 5 switching opportunities, at rounds 3, 7, 11, 15 and 19
* **RCA** — the 4 round types: no switch allowed / switched / chose to stay / stayed but group composition changed
* **RCB** — punishment rate (0, 0.25], (0.25, 0.5], (0.5, 1], > 1, over punished non-full contributors
* **RSA** — punishment 1–3, 4–15, 16+, over punished contributors only
* **RPA** — contribution {0}, 1–5, 6–10, 11–15, 16–19, {20}
* **RPB** — group sizes {1–3, 4–5, 6–8}, over rounds 4 onward; empty groups drop out

### Scoring

All metrics are normalised against a human-vs-human noise ceiling. Over `R ≈ 500` repeats, split the human data into disjoint halves `h_a`, `h_b` of size `m = n_human / 2` and draw a fresh size-`m` sample `s` from the simulation pool. With `d` the row's discrepancy:

`score = E_r[d(h_a, s)] / E_r[d(h_a, h_b)]`

`d` is the row's ★ metric — an absolute difference, or an EMD taken as 1-Wasserstein on the empirical samples without binning. Where the cell is conditional, compute `d` within each stratum `x` and aggregate as `Σ_x w_x · d_x`, with `w_x` the **human** frequency of the stratum. A model matching the human data scores 1.

* ≈ 1 — at the ceiling
* 1–2 — minor deviation
* 2–5 — clear deviation
* \> 5 — not reproduced

Required:

* Both terms computed at size `m` against size `m`.
* Average numerator and denominator separately over the `R` repeats, then divide.
* Reuse the same `h_a` in both terms.
* Split by episode (two groups in a game).
* Fix the split seed, so every candidate model is scored on identical splits and simulation draws.
* Draw all simulation samples from one pre-simulated pool, so no rollouts happen inside the loop.

### Visualisations

Each plot overlays human and simulation, and shows the same object its row's metric measures — so rows scored by a conditional EMD carry a spread band rather than a bare mean curve.

|     | plot | x | y |
|-----|------|---|---|
| CA  | histogram | participant mean contribution | count |
| CB  | lineplot | round | mean contribution |
| CC  | histogram | group mean contribution | count |
| CD  | histogram | contribution | count |
| CE  | histogram | signed group difference | count |
| CE  | lineplot | round | std of the group difference across games |
| CF  | lineplot | round | share at 0 and share at 20 |
| SB  | lineplot | switching opportunity | switch rate |
| SC  | histogram | size of the larger group | count |
| SC  | lineplot | round | size of the larger group |
| PA  | histogram, log y | punishment | count |
| PB  | lineplot | round | mean punishment |
| PC  | lineplot | round | share of punishments at zero |
| RCA | barplot with median and IQR | round type | contribution change |
| RCB | lineplot with median and IQR band | punishment rate stratum | contribution change |
| RCC | barplot | punishment status of full contributors | mean contribution change |
| RCD | scatter with fitted line | `Ĉ − C_n` | `C_{n+1} − C_n` |
| RSA | lineplot | punishment stratum | switch rate |
| RPA | lineplot with median and IQR band | contribution bin | punishment |
| RPB | lineplot with median and IQR band | group size stratum | punishment |

SA is a single rate and has no plot.

## Notes (safe to ignore)

### Considered and decided against

* Mean differences are not reported except where they are canonical. They are one and the same number at CA, CC and CD, ≡ 0 at CE by label symmetry, and elsewhere already visible in the figures — RPA's and RPB's mean curves are the RPA and RPB lineplots, RCA's is the barplot.
* No numeric focal-mass check is reported. EMD is nearly blind to focal structure — smearing the 12% of mass at contribution 0 across 0–2 costs about 0.12 contribution points against CD's floor of 0.97, a score of 1.1 — but the CA, CC and CD histograms show the spikes directly and CF covers the two boundaries numerically. All five focal values spike in the pilot (shares 0.12 / 0.07 / 0.11 / 0.06 / 0.13, against 0.02–0.04 at their neighbours).
* Only RCD is summarised by a fitted slope. The manager's policy is bounded and hinged near contribution 20, and the group-size response is not even monotone, so a linear coefficient describes them poorly while the conditional EMD assumes no functional form. RCD's relationship is genuinely close to linear — the implied pull per gap bin runs 0.32, 0.40, 0.27, 0.63, 0.46 against an OLS coefficient of 0.425 — so the linear form is earned rather than assumed. Binning the gap and scoring mean change per stratum also works but costs power: a model with no pull scores 3.9 that way against 6.4 with the slope, since 539 events pool better than they split. The regression is used rather than the per-event ratio `(C_{n+1} − C_n)/(Ĉ − C_n)`, which diverges whenever `Ĉ ≈ C_n`.
* Punishment within-group and between-group variance are omitted: both are largely implied by PA's spread together with RPA's response curve applied to the observed contribution spread.
* KL and JS were rejected: they ignore the ordering of contributions, diverge on zero support, and would need per-level binning at CA, CC and CE.
* Every alternative to std as the dispersion statistic was tested and rejected. Half-split relative floors on raw contributions / raw punishments / participant means: std 5.6, 10.1, 6.9%; GMD 5.6, 13.5, 7.7%; `E|X − med|` 7.1, 14.7, 8.8%; MAD 10.3, degenerate, 15.4%; IQR 14.0, 32.3, 14.9%. Quantile-based measures fare badly because the distributions have large atoms — contributions at 0/5/10/15/20, punishment at 0 — so quantiles sit on the atoms and move in jumps, and MAD collapses entirely on punishment where the median is zero. The Gini mean difference would be the partner to EMD in the same L1 geometry but buys nothing measurable.
* Round-blocking does not sharpen CB. The floor is set by the 25 independent episodes per half and stays at 0.83–0.90 for every blocking from 24 rounds down to 4, while the trend range shrinks, so the ratio worsens. Only more episodes would help.
* RSA may equally be conditioned on the contributor's contribution gap to the rest of the group; punishment is chosen because it also couples the switching model to the manager.
* RSA over all contributors was rejected in favour of the punished only. The restriction costs raw precision — the floor rises from 0.033 to 0.059 switch-probability as the sample falls to 608 decisions — but nearly doubles the power for what RSA tests. Unrestricted, the zero-punishment stratum carries 70% of the weight while carrying no gradient information, so a gradient error δ moves the score by `0.30·δ / 0.033 ≈ 9δ`; restricted it moves by `δ / 0.059 ≈ 17δ`. SA tests the level, RSA the gradient, with no overlap.
* No confidence interval for now. Scores are comparable across rows as effect sizes, but their precision is not — all rows rest on the same episodes yet differ in episode-to-episode stability (CD stable, CB fragile), so a single score at a fragile row should not be over-read. Model *comparisons* are less affected: with the split seed fixed, candidates share the denominator entirely, so score differences are better determined than the scores themselves. An interval, if added later, needs an outer bootstrap over episodes — the spread of the `R` splits cannot supply it, being conditional on the episodes at hand.
* Energy distance is the analytic alternative to the ratio: it removes the same floor by subtraction, giving an unbiased estimator and a known null distribution, at the cost of the interpretable 1.0 ceiling.

### Implementation details

* The per-stratum discrepancy is the absolute difference of the statistic in columns 1 and 2, and the EMD between the human and simulated distributions of the response variable in column 3. Unconditional cells are the same thing with a single stratum.
* EMD transport is always on the response axis — punishment for RPA and RPB, contribution change for RCA — never on the conditioning variable. Weighting by the human frequency reads as the expected discrepancy for a randomly drawn human observation, damps sparse strata without an arbitrary cutoff, and isolates the manager's policy from how often each situation arises, since contribution levels and group sizes are outcomes of the other two models.
* For a binary response the conditional EMD *is* the frequency-weighted absolute difference in rates: the ground metric on {0,1} gives `W1(Bern(r_h), Bern(r_s)) = |r_h − r_s|`, so columns 1 and 3 coincide at CF, PC and RSA. SA is the same construction with a single stratum.
* Conditioning on round is what pairs two curves round by round rather than comparing them as unordered sets, so CB, PB and PC reduce to `‖curve_h − curve_s‖₁ / n_rounds`. Rounds are equally frequent, so the weights are uniform there; round types, contribution bins and group sizes are not.
* CE uses the signed difference, for which `std(d) = 2 × SD_between-group`.
* RCB's conditioning variable is the punishment rate `p_t / (20 − c_t)` — punishment as a fraction of the shortfall it was levied on — with the response `Δc = c_{t+1} − c_t`. Full contributors are excluded because the rate is undefined for them, and the unpunished because the rate is zero for all of them; 2,697 of the 8,019 eligible observations remain, at 1251 / 679 / 484 / 283 per stratum. RCC's statistic is the contrast `mean Δc among punished − mean Δc among unpunished` over full contributors, and the discrepancy is the absolute difference of contrasts; the contrast rather than the bare mean, so that a model with the wrong ceiling drift is not credited with a punishment response.
* CB and PB compare per-round means rather than per-round distributions: a half-sample gives 200 observations per round but only 25 independent episodes, and shape is already tested pooled at CD and PA and cross-sectionally at CA and CC.
* RPA uses ~6 contribution bins ({0}, 1–5, 6–10, 11–15, 16–19, {20}), isolating the spikes at 0 and 20. This is a sensitivity choice, not a sparsity rescue: measured on the 50-episode pilot (9,600 punishment observations, 4,800 per half), even 21 raw levels drops no strata, the thinnest holding 72 observations per half. Coarser strata lower the noise floor and so raise sensitivity to systematic error, but lose the ability to localise where the policy is wrong — the floor is 0.77 punishment points at 21 levels, 0.60 at 6 bins, 0.46 at 3. At 6 bins a uniform policy offset of 0.5 punishment points scores 1.8 and 1.0 point scores 2.7, against a mean punishment of 1.72. Drop strata with no simulated observations and record the dropped mass.
* RSA uses exponentially spaced strata. Measured on the pilot: 2,000 decisions at rounds 3/7/11/15/19, overall switch rate 0.286, and a clean monotone response — 0.223 at zero punishment rising through 0.291 and 0.535 to 0.720 at 16+. Exponential spacing keeps every stratum above n=50; linear bins leave n=49 and n=37 in the top two and so discard the high-punishment region through down-weighting.
* RPB excludes rounds 0–3. Before the first switch every group is 4/4, so 67.6% of all size-4 observations come from those rounds — which are also when punishment peaks. Unrestricted, size 4 shows a mean punishment of 2.81 against roughly 1.2–1.7 at every other size; restricted to rounds 4 onward the spike vanishes (1.50) and the signal range collapses from 1.65 to 0.52 punishment points. Without the restriction RPB would score models on reproducing a round effect wearing a group-size costume.
* Matched sample sizes matter because EMD's floor scales ~1/√n: a full-size numerator against a half-split denominator scores ~√2 too low. Per-split ratios are biased upward and explode on small denominators. Reusing `h_a` correlates the two terms and lowers the ratio's variance. Unbalanced halves absorb cohort drift into the denominator and flatter every model.

### Other notes

* CF exists because the mean hides the dynamics. Human contributions are nearly flat over the 24 rounds (8.16 → 9.74) while the population polarises: the share at 0 rises 0.085 → 0.160 and the share at 20 rises 0.085 → 0.180. CD pools rounds and so sees only the time-averaged mixture, and CB sees the flat mean, so a model producing a static distribution with the right average would score clean at both. CF is also the sharper temporal test, its floor being 34–40% of the trend range against CB's 54%. The two boundaries stay separate terms in the aggregate, so a model that swaps mass from 0 to 20 cannot score clean.
* RCB conditions on the rate rather than on the punishment amount, because the amount is confounded with the contribution level. Mean contribution change by raw punishment stratum runs −0.39, +0.53, +1.16, +2.66, but mean contribution across those same strata falls 10.2 → 3.5, so most of the gradient is low contributors reverting upward and a model with correct mean-reversion and no punishment sensitivity would score near 1. Normalising by the shortfall removes it, and better than neutrally: across the rate strata the human response rises +0.84, +1.29, +1.42, +1.72 while mean contribution *rises* 6.14 → 7.45, which predicts less upward reversion, so mean-reversion works against the observed gradient and any gradient detected is a lower bound.
* RCB's floor is 0.345 contribution points against a no-effect discrepancy of 1.205, so a model in which punishment does nothing scores about 4.5. Alternatives measured: the same strata scored by conditional EMD reach 3.0, a punished-minus-unpunished contrast within contribution bins 2.8, and the raw punishment strata about 1.0.
* RCB is scored on the mean rather than the EMD because the punishment effect is a modest location shift inside a wide distribution: across the four rate strata the mean runs +0.84, +1.29, +1.42, +1.72 against standard deviations of 3.38, 3.98, 5.47, 6.88. A no-effect model scores 4.5 on the mean and only 3.0 on the EMD. The response does have shape the mean cannot see — 21–31% do not move at all, and the share moving *downward* rises from 21% to 28% as the rate grows, so harsher punishment buys both more compliance and more retaliation — but change-distribution shape is already tested at RCA and CD, while the dose-response is RCB's alone. The figure carries the spread.
* RCC covers punished full contributors, who fall outside RCB because the rate is undefined at a zero shortfall. They drop 9.4 contribution points against −1.9 for unpunished full contributors, giving a floor of 4.29 against an effect of −7.45, so a model without the response scores about 2.7. The row is thin and provisional: 47 observations from 14 episodes and 25 participants, concentrated early (median round 6 against 13 for full contributors generally), and bimodal rather than a shift — a quarter drop from 20 straight to 0 while the upper quartile does not move.
* PC is the extensive margin of punishment. PB is the product of the two margins and cannot separate them, and in the human data the taper is almost entirely extensive: the share punished falls 0.505 → 0.167 across the 24 rounds (−67%) while severity among the punished falls only 7.37 → 5.97 (−19%). A model reproducing PB's curve through the intensive margin instead — the same fraction punished, progressively softer — would be a qualitatively different manager and would match a pooled zero share as well. Conditioning on round costs a little sensitivity to a pure level error (floor 0.054 rather than 0.036) and buys a trend spanning 0.495 → 0.833 at 16% of its range.
* PC also anchors RSA, which is computed over punished contributors only: without it nothing checks that the model punishes the same fraction of people. PA's EMD is too soft to serve — shifting 11% of punishments off zero costs about 0.22 punishment points against PA's floor of 0.30, so a score near 1.7 for a large policy difference.
* CB and PB differ sharply in power. PB's floor is 0.46 punishment points against a trend spanning 1.00–3.72, so 17% of the signal — a sensitive test of the manager's punish-early, taper-late profile. CB's is 0.86 contribution points against a trend spanning only 8.16–9.74, so 54%: contributions barely move, because the institution sustains cooperation rather than letting it decay. CB is a guardrail against gross trajectory failure, not a discriminator between good models.
* An underpowered row is not a misleading one: if data is thin the denominator grows in step and the score sits near 1, so power is lost rather than false alarms gained. Without an interval, though, a score of 1.2 at CB or PB cannot be distinguished from "nothing was detectable" — those two rows are where an interval would earn its keep first.
* RPB is a guardrail, not a fidelity test. After the round restriction its floor is 0.50 punishment points at raw sizes, 0.36 at {1–3, 4–5, 6–8} and 0.32 at {1–4, 5–8}, against a signal range of 0.52 — so it cannot resolve the human group-size effect, because there barely is one. Its remaining value is catching a model that invents a group-size dependence humans do not have, and the coarse three-stratum binning is chosen on that basis.
* SA and SC are not redundant: the same switch rate can leave groups balanced or fully segregated depending on directionality.
* `Δstd` adds no detection power over the EMD — any spread error already shows up there — but it is signed and reportable in raw units. EMD is a distance and cannot say whether a model is too tight or too loose, and under-dispersion is both the characteristic failure of networks trained on cross-entropy or MSE and the error hardest to see in a histogram.
* The ruler is a property of the human data, so scores are comparable across candidate models.
