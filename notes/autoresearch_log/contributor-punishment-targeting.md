# Does punishment mean anything different to a full contributor? An interventional probe of the frontier contributor

## 1. The question

Three RL-manager seeds trained against the current clone stack came out punishing the wrong people: two of three punish full contributors hardest and leave free-riders alone, the inverse of the human policy, which falls from 4.76 at contribution 0 to 0.27 at contribution 20 (PR #204, `notes/autoresearch_log/rl-manager-two-worlds.md`). Four exploration arms are training now on the hypothesis that action-level exploration decorrelates punishment from contribution in the replay buffer and leaves the learned contingency arbitrary.

None of those four can rule out a rival explanation. **If the artificial contributor responded to punishment without regard to what it contributed, no manager would have any gradient toward correct targeting**: the inversion would cost nothing in behaviour, the four arms would differ only on level and seed spread, and none of them would be measuring what it thinks it is.

The evaluation suite already names the human quantity. **RCE** is the OLS slope of the next-round contribution change on the punishment received, within contribution band; on the human data it runs **+0.140 / +0.104 / -0.077 / -0.161** across the bands 0-4 / 5-9 / 10-14 / 15-19, with the human-vs-human noise ceiling at **0.086**. The sign reverses, and that reversal is the whole reason a manager facing people should aim punishment low and not high.

RCE as the suite computes it is **observational**, and cannot settle this. The punishment an agent received was chosen by a manager that had already looked at the contribution, so a within-band dose slope mixes the causal response with whatever else the manager conditioned on. This probe asks the causal question instead, on the contributor model alone, with no reinforcement learning anywhere in it: take real states, **force** a contribution and **force** a punishment, and read the model's next-round contribution.

## 2. Setup

**Model.** The frontier contributor the four RL arms train against, byte for byte the artifact named in `configs/training/rl_manager/rl_new_clones_s4*.yml`: `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` (`group_vnode: True`, `stimulus_skip: True`, `copula_rho = 0.03949863621805423`, `copula_phi = 1.0`; `x_encoding = prev_contribution, prev_punishment, agent_group`). Nothing is trained, recalibrated or written to `artifacts/`.

**Data and contexts.** The canonical 50 single-copy human episodes (`experiments/2group_8agent_50ep.csv` through the training loaders, the flip copies deduplicated exactly as `contribution_copula_rho.load_full` / `select_split` do it). A context is an (episode, focal agent, round t*) triple with t* in {5, 9, 13, 17, 21}, kept only where the focal agent's own round-t* cell was recorded: **1,885 contexts**.

**The intervention.** For every context and every cell of the **21 x 31 = 651**-cell grid (contribution 0-20 x punishment 0-30), the focal agent's round-t* contribution is forced to `c` and its punishment to `p`, in both the round-t* slot and the `prev_*[t*+1]` slot the model reads, with `common_good[t*]` recomputed from the overridden values and propagated to `prev_common_good[t*+1]` — the same override `simulation/intervention_probe` performs. The response is the model's own conditional mean, `delta(c, p) = E[c_{t*+1}] - c`. Every context sees every cell, so the surface is paired and context heterogeneity cancels in the slopes.

**Shared noise: OFF, and it cannot matter.** `predict_independent(sample=False)` returns the predicted marginal and draws nothing, so no RNG is consumed and the herding copula is never entered. The copula is an inverse-CDF sampler that correlates draws across the agents of a cell while leaving each row's marginal exactly as predicted, so the measured quantity is invariant to rho and phi by construction. That invariance is **measured, not asserted**: nine cells redrawn with the copula on, 20 draws each (4,200 draws per cell), give sampled means within **max |z| = 0.885** of the analytic ones (`copula_invariance.csv`). The one place shared noise is switched on is the rollout of §3.5, where draws are needed and the paired seeds make it work for the measurement rather than against it.

**The imputed 9.** A player who gave no input is stored with the median contribution 9 and `contribution_valid = False`. Those cells are excluded from the human reference population and from the context set, and the model's outcome is a distribution over valid contributions (the trunk trained under `mask_name: contribution_valid`), so no imputed value is averaged in anywhere.

**Two checks before any number is read.**

1. The rebuilt human RCE population (n = 2,660) reproduces the recorded human slopes to four decimals: +0.13974110 / +0.10384679 / -0.07665764 / -0.16149422 against the frozen +0.1397 / +0.1038 / -0.0767 / -0.1615.
2. **Forcing a context's own recorded cell is a no-op.** Set `(c, p)` to what the round really was and the forced response must equal the plain teacher-forced prediction for that row — a different forward, over all 24 rounds instead of the probe's truncated prefix, on the cpu instead of the gpu, through `contribution_copula_rho.teacher_forced_rows`. Over the 1,868 qualifying contexts: **max |difference| = 0.0031**, mean 0.0002 contribution points. Both are float32 arithmetic over different sequence lengths on different devices; 0.0031 is 1.6e-4 of the 0-20 scale and two orders below the smallest effect read off the surface. That is what licenses reading every other cell of the grid as the model's answer rather than as an artefact of how the override was written.

**Reproduction.** Three jobs on one A100: 28 s, 53 s and 2 min 32 s; nothing else on the cluster was touched.

```bash
# the surface, the band slopes and the copula invariance check
sbatch scripts/data_analysis/contributor_punishment_intervention.slurm
# the own-path rollout
sbatch scripts/data_analysis/contributor_punishment_rollout.slurm \
    --n-contexts 900 --n-draws 12 --horizon 8 --dose 8
# the seed spread: the same probe with --model <seed artifact> --output-dir <dir>,
# then, locally
python scripts/data_analysis/contributor_intervention_seed_spread.py \
    <shipped>/summary.json <seed_*>/summary.json --out <evidence dir>
# the figures, locally
python scripts/data_analysis/plot_contributor_intervention.py
```

The five seed artifacts live on `origin/auto/seed-spread-noise-floor` under `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_seed_ensemble_copula/` and were read from that ref rather than committed here, since they belong to that branch.

## 3. Results

### 3.1 The intervention surface (measured)

`plots/data_analysis/contributor_punishment_targeting/intervention_surface.csv`, figure `intervention_surface.jpg`. The punishment effect proper — the forced response minus the same contexts' response at zero punishment, in contribution points:

| forced c \ forced p | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.013 | -0.027 | -0.040 | -0.062 | -0.073 | -0.028 | +0.239 | +0.698 |
| 2 | +0.035 | +0.070 | +0.106 | +0.178 | +0.293 | +0.465 | +0.907 | +1.439 |
| 5 | +0.070 | +0.141 | +0.213 | +0.356 | +0.569 | +0.840 | +1.210 | +1.353 |
| 8 | +0.034 | +0.071 | +0.112 | +0.198 | +0.304 | +0.384 | +0.464 | +0.490 |
| 11 | +0.079 | +0.151 | +0.211 | +0.291 | +0.314 | +0.227 | +0.009 | -0.081 |
| 14 | +0.017 | +0.029 | +0.035 | +0.028 | -0.022 | -0.131 | -0.365 | -0.589 |
| 17 | -0.002 | -0.007 | -0.015 | -0.042 | -0.107 | -0.226 | -0.493 | -0.809 |
| 19 | -0.003 | -0.012 | -0.027 | -0.073 | -0.180 | -0.367 | -0.741 | -1.117 |
| 20 | -0.013 | -0.032 | -0.057 | -0.124 | -0.260 | -0.488 | -0.938 | -1.359 |

**The shape is human.** Compliance at low contributions, withdrawal at high ones, a sign flip in between (near c = 12). **The size is not.** A human contributing 17 loses 0.161 contribution points for every punishment point; this model needs a dose of about 25 to lose that much.

### 3.2 The RCE analogue, forced (measured)

`band_slopes.csv`, figure `band_slopes.jpg`. Weighted by the **human** cell counts, so the model's interventional slope and the human's observational slope are computed over the same cells with the same composition and differ only in where the response comes from.

| band | human (observed) | **model (forced)** | bootstrap 95% CI | near-support | whole dose range | share of human |
|---|---|---|---|---|---|---|
| 0-4 | +0.1397 | **+0.0717** | [+0.0702, +0.0732] | +0.0604 | +0.0482 | 51% |
| 5-9 | +0.1038 | **+0.0625** | [+0.0607, +0.0642] | +0.0556 | +0.0230 | 60% |
| 10-14 | -0.0767 | **+0.0105** | [+0.0082, +0.0126] | +0.0141 | -0.0161 | -14% |
| 15-19 | -0.1615 | **+0.0057** | [+0.0037, +0.0073] | +0.0346 | -0.0325 | -4% |

**The decisive number.** The targeting gradient, the 0-4 slope minus the 15-19 slope — what one punishment point buys when it is moved from a full contributor to a free rider:

| | value |
|---|---|
| human, observed | **0.3012** |
| model, forced, human cell weights | **0.0660**, bootstrap 95% CI [0.0640, 0.0683] |
| **share of the human gradient that survives** | **21.9%** |
| model, forced, within contribution level (§3.2b) | 0.0589 against a human 0.2696 — **21.8%** |
| model, forced, near-support contexts only | 0.0258 (8.6%) |
| model, forced, whole 1-30 dose range | 0.0807 (26.8%) |
| over six training seeds (§3.4) | mean 0.0607, sd 0.0121, **5.0 seed sd from zero** |
| model mean slope over the four bands (the LEVEL) | +0.0376 |
| human-weighted mean abs slope gap, in RCE noise ceilings | 0.819 |

The gradient is **not zero**: the context bootstrap puts it **58 standard errors** clear of zero (sd 0.00114), the level-by-level curve (`level_slopes.csv`) is orderly rather than noisy — it rises from +0.022 at c = 0 to a peak of +0.066 at c = 4, declines through zero between c = 11 and c = 12, and settles between -0.014 and -0.032 from c = 12 to c = 19 — and it survives retraining at 5.0 seed sd (§3.4).

### 3.2b The same slope with the level controlled (measured)

RCE fits one line per band, so its slope also absorbs the between-level gradient inside the band: the response falls about three contribution points across 15-19, and if the punishment humans gave differs across those five levels the pooled fit reads that as dose. Demeaning punishment and response *within* contribution level removes it. Both are in `band_slopes.csv`; the pooled statistic is the one the suite scores and the one §3.2 leads with.

| band | human, within level | model, within level | share |
|---|---|---|---|
| 0-4 | +0.1217 | +0.0344 | 28% |
| 5-9 | +0.1038 | +0.0396 | 38% |
| 10-14 | -0.0752 | +0.0018 | -2% |
| 15-19 | **-0.1479** | **-0.0245** | **17%** |
| gradient | 0.2696 | 0.0589 | **21.8%** |

Two things change and one does not. The **gradient is the same** — 21.8% against the pooled 21.9%, so the answer does not come from the pooling. And the 15-19 band recovers its human **sign**: once the level is controlled, the model does withdraw when a high contributor is punished, at about **a sixth** of the human size. The frozen pooled statistic does not show that, because the composition inside the band cancels it.

### 3.3 The ceiling, which RCE drops and the inverted managers aim at (measured)

RCE excludes contribution 20 because its *rate* is undefined there. Its *slope* is not, and contribution 20 is exactly where `rl_s42` and `rl_s44` put their punishment (5.000 and 2.000 mean, PR #204 §2).

| | human | model (forced) | ratio |
|---|---|---|---|
| punished-minus-unpunished contribution change at c = 20 (the RCC contrast) | **-7.035** (n = 1,140; 44 punished) | **-0.246** | 3.5% |
| slope of the change on the punishment at c = 20 | **-0.697** | **-0.049** | 7.0% |

Real full contributors punished at the mean human dose for that level (6.98) collapsed by seven contribution points. This model gives up a quarter of one point. **Punishing a full contributor is, behaviourally, almost inert here.**

### 3.4 Does it survive a retrain? (measured)

The same probe, unchanged, on the five-seed ensemble of the same architecture (`group_switching_contribution_50ep_vnode_stimulus_skip_seed_ensemble_copula`, PR #195's arms) plus the shipped model: six arms differing only in the training draw. `seed_spread.csv`.

| arm | 0-4 | 5-9 | 10-14 | 15-19 | gradient | ceiling contrast |
|---|---|---|---|---|---|---|
| shipped | +0.0717 | +0.0625 | +0.0105 | +0.0057 | 0.0660 | -0.246 |
| seed_1 | +0.0792 | +0.0597 | +0.0190 | +0.0285 | 0.0507 | -0.026 |
| seed_2 | +0.0585 | +0.0472 | -0.0070 | -0.0066 | 0.0651 | -0.504 |
| seed_3 | +0.0845 | +0.0499 | +0.0084 | +0.0097 | 0.0748 | -0.031 |
| seed_4 | +0.0710 | +0.0494 | +0.0265 | +0.0292 | 0.0417 | -0.320 |
| seed_5 | +0.1023 | +0.0637 | +0.0263 | +0.0367 | 0.0656 | -0.005 |
| **mean** | **+0.0779** | **+0.0554** | +0.0139 | +0.0172 | **0.0607** | -0.189 |
| **sd** | 0.0148 | 0.0073 | 0.0128 | 0.0168 | **0.0121** | 0.2024 |
| human | +0.1397 | +0.1038 | -0.0767 | -0.1615 | 0.3012 | **-7.035** |

**The gradient is 5.0 seed standard deviations from zero** (0.0607 / 0.0121), and its share of the human gradient is stable: 20.1% on average, range 13.9% to 24.8%. The 0-4 slope is positive in all six arms. So the dependence is a property of the architecture, not of one draw, and it is not the retraining noise this project measured at a typical row sd of 0.138 and an RCE score sd of 0.106 (PR #195) — those are sds of *scored rows*, and this is the statistic itself, measured on its own arms.

Two things are equally stable and equally far from human. **No arm's 15-19 pooled slope comes anywhere near the human -0.1615** — the six span -0.0066 to +0.0367. And **no arm's ceiling contrast comes within a factor of 14 of the human -7.035** — the six span -0.005 to -0.504. Whatever a different training draw buys, it does not buy the withdrawal response.

### 3.5 Does the one-round response compound? (measured, own-path closed loop)

A one-round answer can understate what a manager is paid on, so the open-loop result was followed by an **own-path rollout**: the same contexts and the same forced cell at t*, then the focal agent's own contribution drawn by the model and fed back into its own history for eight further rounds, with the peers' contributions and every punishment after t* held at the human record so that they cancel between arms and no manager reacts to the perturbation. **Shared noise is ON here** — the rollout needs draws, so `sample=True` runs the model's own copula; the treated and control arms are seeded identically with an identical batch layout, so they share every uniform. The pairing cuts the one-round standard error by **5.9x to 9.5x** against the same difference computed from independent arms (`rollout_by_level.csv`).

Dose 8, gamma 0.98, 900 contexts, 12 draws (`rollout_by_level.csv`, `rollout_paths.csv`):

| forced c | one round | discounted sum over 8 rounds | multiplier | **per punishment point, over the stream** |
|---|---|---|---|---|
| 2 | +0.365 | +0.668 | 1.83x | **+0.084** |
| 7 | +0.370 | +1.006 | 2.72x | **+0.126** |
| 12 | +0.101 | -0.002 | — | -0.000 |
| 17 | -0.111 | -0.864 | 7.76x | **-0.108** |
| 20 | -0.274 | -1.205 | 4.39x | **-0.151** |

**The response compounds rather than decaying, and it compounds hardest on the withdrawal side**, so the open-loop reading is conservative about the shape, not generous. It is still small: summed over eight discounted rounds the model returns **+0.084** contribution points per punishment point at contribution 2 and **-0.108** at contribution 17 — neither reaches what a person delivers in the **single** next round (+0.140 / -0.161).

The own-path design severs the peer channel, so these are lower bounds on magnitude and say nothing about the absolute return to punishing, which the rule sweep has already measured to be strongly positive in the full loop (PR #207: `prop10` 136.04 and `thr9_p10` 123.79 against `never` 99.63).

### 3.6 How much of the observed response is caused by the punishment? (measured)

The same trunk, teacher-forced on the same 50 episodes through the independent step-0 script `scripts/data_analysis/rcb_teacher_forced.py` (alignment checks pass: the lag profile peaks at the conditioning lag, corr 0.9332, and the stickiness slope is 0.7646 against the human 0.7686):

| band | human, observed | model, **observational** | model, **interventional** |
|---|---|---|---|
| 0-4 | +0.1397 | +0.1482 | +0.0717 |
| 5-9 | +0.1038 | +0.1038 | +0.0625 |
| 10-14 | -0.0767 | +0.0171 | +0.0105 |
| 15-19 | -0.1615 | **-0.2197** | **+0.0057** |
| gradient | 0.3012 | **0.3679 (122% of human)** | 0.0660 (22%) |

**Observationally the model looks more human than human; causally it is a fifth.** 82% of its own apparent punishment-response spread is the state the punishment co-occurred with, not the punishment. The entire high-band withdrawal — -0.2197, larger in magnitude than the human -0.1615 — survives none of the intervention.

## 4. Verdict

**Of the three outcomes the question posed, the data shows the third: something in between, at the weak end. 21.9% of the human targeting gradient survives the intervention — 0.0660 against 0.3012 — with a context bootstrap 95% CI of [0.0640, 0.0683], and 20.1% on average over six training seeds (range 13.9% to 24.8%, the gradient 5.0 seed sd from zero).**

The rival explanation is **rejected**. The artificial contributor's response to punishment does depend on what it contributed, causally, in the human direction, in every one of six independent retrains: punishment raises a free-rider's next contribution and — once the level is controlled — lowers a high contributor's. A manager training against this contributor is not optimising in a world where targeting is free.

But the gradient it is handed is about a fifth of the human one, and the half of the human mechanism that makes misaimed punishment actively destructive is the weaker half: a sixth of human within the 15-19 band (-0.0245 against -0.1479), and **3.5% of human at the ceiling**, where the RCC contrast is -0.246 against the human -7.035 and no retrain gets past -0.504. Punishment aimed at a full contributor is not free in this world — it still costs the pool what it costs — but it is very close to behaviourally inert, and contribution 20 is exactly where `rl_s42` and `rl_s44` aim. PR #204's own table says the same thing from the other end: `rl_s42` spends a mean punishment of 1.99 on high contributors and reaches a mean contribution of 8.023, against `never`'s 8.015. The misaimed punishment bought 0.008 contribution points.

**What this means for the four arms, plainly: they are testing something real, and they should not be stopped.** The signal they need a manager to find exists, is correctly signed, compounds over rounds, and survives retraining at five seed sd. What the probe adds is its size — a fifth of the human gradient, against reward noise that PR #204 measured at a three-seed range of 3.401 on a common-good gap of 2.332. If all four arms come back differing on level and seed spread with none recovering the human policy shape, **that outcome is now explicable without any further experiment**: the gradient is real but small, and small relative to the noise the three seeds already showed. It would not need a fifth exploration method to explain it.

## 5. Notes

1. **Measured against inferred.** Sections 3.1-3.6 are measurements on the shipped artifact and, in §3.4, on five retrains of it. Section 4's reading of what they mean for the four arms is inference, and it rests on one step that this probe does not measure: that a manager's learning signal scales with the causal targeting gradient. Nothing here measures a manager.
2. **The human benchmark is observational and there is no interventional human.** Comparing a model's forced slope to a human's observed slope is not like for like, and it is the one asymmetry that could change the reading: if the human +0.140 / -0.161 is itself mostly the state the punishment co-occurred with, then the model is faithful and the whole world — human and simulated — offers a manager less targeting gradient than RCE suggests. That cannot be settled from these data. It does not change §4's answer for the arms, because the arms train against the model, and the model's causal response is what their gradient is made of, whatever people do.
3. **The pooled band slope is not a within-level dose response, and the difference matters for exactly one band.** RCE fits one line per band, so its slope also absorbs the between-level gradient inside the band (the response falls about three contribution points across 15-19). The pooled 15-19 slope of +0.006 is that composition effect: every individual level in the band has a *negative* forced slope (-0.014 to -0.032, `level_slopes.csv`), and the fixed-effects version is -0.0245 (§3.2b). The gradient is unchanged at 21.8%, so nothing in the verdict turns on it, but "the model has no withdrawal response" would be the wrong sentence — "the model's withdrawal response is a sixth of human and the frozen statistic cannot see it" is the right one.
4. **The near-support restriction moves the answer down, not up.** Asked only about contexts whose own round-t* contribution fell in the band being probed, the gradient is 0.0258, 8.6% of human. So the "you are asking the model about states it never reaches" objection does not rescue the size of the response; on the manifold the model actually visits, the gradient is smaller still. This is also the number closest to what a manager faces, since a manager punishes a player at the contribution the player chose.
5. **The extensive margin runs the other way at the ceiling.** The step from no punishment to one point is positive at almost every contribution level (`model_step_p0_to_p1` in `level_slopes.csv`, +0.03 to +0.08 in the middle of the range), while the dose gradient beyond it turns negative above c = 12. That is why the human-weighted 15-19 slope (doses mostly 1-5, mean 3.80) comes out near zero while the whole-range slope is -0.032. Humans respond negatively to both at that level.
6. **A free-rider at exactly zero is the least responsive low contributor.** The forced slope at c = 0 is +0.022, against +0.066 at c = 3-4, and small doses at c = 0 move the response slightly *down* (-0.013 at p = 1, -0.073 at p = 8) before turning up at p >= 20. Whatever a manager learns about punishing hard-core free-riders in this world, it will not be learned from a small dose.
7. **The no-op check is the probe's real guarantee**, and it is in §2 rather than the results because it is a precondition for reading any of them.
8. **The copula was checked, not assumed.** rho = 0.0395 and phi = 1.0 are live on this artifact, and the probe's estimand is invariant to them because the sampler preserves each row's marginal exactly. Nine cells redrawn with the sampler on agree with the analytic surface at max |z| = 0.885.
9. **The rollout's multipliers are not uniform, and the asymmetry favours the withdrawal side** (1.83x at c = 2, 7.76x at c = 17). A probe that measured only the stream would therefore report a *larger* share of the human gradient than the one-round probe does. The one-round number is the one quoted in §4 because it is the estimand RCE defines; the stream is reported because a manager is paid on it.
10. **What did not happen.** Nothing was trained, no artifact was written, no simulation config was added, the four running arms' directories were never touched, and the remote work lived in `~/autoresearch/contributor-punishment-targeting`.

## 6. For a successor

1. **The decisive missing measurement is the closed-loop cost of the inversion, and it is one simulation.** The rule-sweep harness on `auto/rule-based-manager-sweep` (`RuleBasedManager` family, `scripts/data_analysis/rule_manager_configs.py`) already contains `thr9_p10` — punish 10 whenever the contribution is at most 9. Add its mirror, `inv_thr11_p10` — punish 10 whenever the contribution is at least 11 — and run it beside `thr9_p10` and `never` in one config, on the standard protocol. The three numbers answer directly what §3 answers only for one agent and one round: what does aiming punishment at the wrong people cost, in the pool, in the full loop. This probe predicts the inverted rule lands close to `never` on contribution and below it on the pool by roughly the punishment it spends. That branch is not merged into the tree the RL arms launched from, which is why it was not run here.
2. **RCE cannot tell a causal response from a confounded one, and it is a protected row.** The shipped contributor scores RCE 0.882, the only arm of six at or under the noise ceiling (PR #195), while its causal response is a fifth of human. Any future contributor experiment that targets the punishment response should report the forced slopes beside the scored ones; `scripts/data_analysis/contributor_punishment_intervention.py` does it in under half a minute on one GPU. A candidate that improves RCE by improving the state channel has not improved the mechanism a manager uses.
3. **The ceiling deserves its own row.** RCE drops contribution 20 because the RCB rate is undefined there, and RCC scores the ceiling only as a punished-minus-unpunished contrast without a dose. The model is 3.5% of human on that contrast and 7.0% of human on the slope, and contribution 20 is where the inverted managers aim. A ceiling dose-slope row would have caught this without an intervention probe.
4. **If the arms come back flat, the next question is not a fifth exploration method.** It is whether a manager can learn a gradient this size at all: fit the manager's own Q-function against the forced surface of §3.1 and ask whether the action-value difference between punishing a 2 and punishing a 17 is resolvable against the replay noise. That is a measurement on the trained arms and costs no new training.
