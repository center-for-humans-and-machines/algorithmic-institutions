# The contribution copula recalibrated on top of the serving-path fix

## 1. Declaration

**Slot:** contribution -- and specifically the one frozen noise parameter this experiment is declared to unfreeze, the contribution copula's correlation strength `rho`. §2 (amended, `docs/post-rebaseline-program`) freezes `rho` and `phi` per model family and says that altering either "is its own declared experiment"; this is that experiment for `rho`. **The persistence `phi` is not touched** -- it stays at the stamped 1.0, it is a separate frozen parameter and a separate question.

**Nothing is retrained.** No weights, no features, no training configs. The contributor trunk, the switch model, the punisher and the validity model are the same files on disk the parent ran; the only thing that may change is three scalar fields stamped onto a copy of the contributor artifact.

**Parent:** `auto/sim-timeout-imputation` at `3fe1f44` (the serving-path fix: the contribution and switch models are served the recorded 0 for a timed-out player instead of the imputed 9, on 2.24% of agent-rounds). Branch `auto/contribution-copula-recalibrated` is created from it and the PR opens with `--base auto/sim-timeout-imputation`. Isolated remote dir `~/repros/ai-runs/copula-recal`; delete it when this PR closes.

**Base models (unchanged, not retrained).** Contributor trunk `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` (sha256 `9de0d772...`, the same file the shipped calibration was fitted on), stamped frontier contributor `..._vnode_stimulus_skip_herding_copula/...` (sha256 `ada5d2aa...`), switch `artifacts/artificial_humans/switch_joint_exodus/...`, punisher `artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib`, validity `artifacts/artificial_humans/raven_script_22/...`.

**Evaluation stack (§3 under the parent rule of §9).** The frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch`, re-run from a config that differs from the parent's `_simtimeout` in `output_dir` and `figure_name` and in the contributor artifact and in nothing else (`_copularecal`).

**Baseline (the parent's confirmed `_simtimeout` frontier scores).**

| row | score | band |
|---|---|---|
| mean over 22 rows | **1.0393** | |
| gate-2 ceiling (mean x 1.10) | **1.1432** | |
| rows <= 1 | 14/22 | |
| **CG (declared target)** | **1.8449** | 1-2 |
| SC | 1.8295 | 1-2 |
| RCC | 1.4983 | 1-2 |
| RCE (protected) | 0.9474 | <= 1 |

### Hypothesis

**The claim.** The parent's measurement was that its serving fix is right and that its score cost is confined to dispersion rows on the one stack whose contributor carries a copula: with the shared-noise machinery off the group-spread ratio is flat across the fix (0.7837 -> 0.7815) and the state spread rises (18.92 -> 21.36 against the human 27.93), while with the machinery on CG falls further below the human and the reference stack, whose contributor carries no copula, *improves*. The reading offered there (note 4) is that the mis-served nines were an accidental source of between-group dispersion that the copula's calibration had absorbed. **If that is right, a `rho` fitted against a model that no longer sees those nines should come out larger than the shipped 0.0395, by more than the estimator's own bootstrap spread, and stamping it should recover CG.**

**Behavioural rationale (§5, one sentence):** the copula exists to supply the between-participant correlation that independent per-agent sampling cannot (the CG/PD/SC root cause of §6), and its strength should be the strength measured against the model as it is actually served -- the row that should move is CG, the group-spread ratio.

**The change.** One scalar: `copula_rho` on a copy of the frontier contributor artifact. `copula_phi` stays 1.0, `copula_switch_every` stays 1, every weight stays bit-identical.

**Declared target row: CG.** Gate 1 is a band improvement on CG (1-2 -> <= 1) that also exceeds CG's seed sd of 0.301; gate 2 is the 22-row mean at or below 1.1432. SC and RCC are reported as watch rows (both lost legibly on the parent), RCE is the protected row under the amended rule.

### Artifact naming contract

| what | path |
|---|---|
| refit params | `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula_recal/calibration/copula_params.json` |
| stamped artifact | `..._herding_copula_recal/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| calibration job | `scripts/artificial_humans/calibrate_copula_stimulus_skip_recal.slurm` |
| stamping job | `scripts/artificial_humans/stamp_copula_stimulus_skip_recal.slurm` |
| sim config (gated) | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_copularecal.yml` |
| sim config (noise-off) | `configs/simulation/manager_testing/23_2g8a_copula_recal_rho0.yml` |
| state-spread diagnostic | `scripts/data_analysis/copula_recal_closed_loop_variance.py` |
| tables | `plots/data_analysis/evaluation/contribution_copula_recalibrated/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Establish what the estimator's inputs actually are on the corrected tree: which files it reads, whether any of them moved with the serving fix, and the hashes of the trunk and the human data it was last fitted on. | **done** -- section 3, step 1 |
| 2 | Refit `rho` with the existing estimator (`contribution_copula_rho.py`, same flags, same seed) against the corrected tree; report the estimate, its bootstrap CI and SE, and whether the move from 0.0395 is inside the estimator's ordinary spread. | **done** -- job 30326043 |
| 3 | Stamp the refitted value onto a copy of the contributor artifact and verify the copy is weight-identical to its base outside the three copula fields (dict-level bit comparison plus the teacher-forced probability check the stamping script already runs). | **done** -- job 30326957 |
| 4 | Simulate the frontier stack with the recalibrated contributor and evaluate all 22 rows against the parent's, every movement quoted beside its PR #195 seed sd. | **done** -- job 30326959 |
| 5 | Run the noise-off arm and the state-spread diagnostic; record what the recalibration does and does not change about `Var(E[c | history])` and the group-spread ratio. | **done** -- jobs 30325909 / 30327026 |
| 6 | Check the protected row RCE under the amended rule (band-drop threshold 0.106, sign clause retired on 10-14 and 15-19), reporting every band slope with its standard error and row count. | **done** -- section 3, step 6 |
| 7 | Judge, log, PR against `auto/sim-timeout-imputation`. | **done** -- section 5 |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | the contribution copula's `rho` refitted against the corrected serving path (nothing retrained) | **CG 1.8449 -> 1.8449** (delta **exactly 0**, 0.00 seed sd of 0.3014, **not distinguishable**). Watch rows: SC 1.8295 -> 1.8295, RCC 1.4983 -> 1.4983, both delta 0. | 14 -> **14/22** (0) | 1.0393 -> **1.0393** (gate-2 ceiling 1.1432, pass; 0.00 seed sd) | **[FAIL]** -- no band upgrade, because nothing moved at all; the hypothesis is **refuted** (section 5) |

**The refit returned the shipped value to the last digit and the stamped artifact is the shipped file byte for byte, so every number below is an equality, not a small difference.** The chain was run end to end anyway, and each link was measured rather than assumed.

### Step 2: the refit (measured -- Raven job 30326043, 10 min 36 s)

`scripts/artificial_humans/calibrate_copula_stimulus_skip_recal.slurm` runs the existing estimator on the corrected tree, same base trunk, same flags (`--roundtrip --preflight`), same seed 38381 as the job that produced the shipped value.

| quantity | shipped (2026-09-16) | refit (2026-09-19) | change |
|---|---|---|---|
| **`rho`** | **0.03949863621805423** | **0.03949863621805423** | **exactly 0.0** |
| bootstrap SE | 0.00984156677188504 | 0.00984156677188504 | 0 |
| 95% percentile CI | [0.018089565319783816, 0.056620768540509354] | [0.018089565319783816, 0.056620768540509354] | 0 |
| round-trip max bias (acceptance gate) | 0.006643595460441576 | 0.006643595460441576 (PASS, tol 0.03) | 0 |
| holdout `rho` | 0.08510866692268397 | 0.08510866692268397 | 0 |
| `rho_lag1` cross-player / `phi` | 0.03208097178376507 / 0.8122045431305637 | identical | 0 |
| pre-flight group-spread (indep / copula / human) | 0.8357 / 0.8415 / 0.8473 | identical | 0 |
| randomized-PIT diagnostic | 0.024882282260142934 | 0.024882282260142934 | 0 |
| rows / cells >= 2 / pairs | 7457 / 1608 / 15090 | 7457 / 1608 / 15090 | 0 |

A field-by-field comparison of the two params JSONs finds **all 36 estimate and provenance fields identical**; only `date` and `git_head` differ.

**Is the change inside the estimator's ordinary spread?** It is not a change. The move is 0.0000 against a bootstrap SE of 0.0098 -- **0.00 SE** -- and against a 95% CI of width 0.0385. There is no version of this comparison in which a zero move is evidence for a recalibration.

### Step 3: the stamped artifact (measured -- Raven job 30326957, 22 s)

`make_contribution_copula_artifact.py` copied the base trunk and stamped `copula_rho = 0.03949863621805423`, `copula_phi = 1.0`, `copula_switch_every = 1`:

- **14 tensors compared bit-identically** against the base artifact, every pre-existing key verified unchanged after reload;
- **7457 teacher-forced train-split rows** bit-identical to the base model's predicted probabilities (the honesty check);
- output sha256 **`ada5d2aa8f2fe71bcf63c9da0274f572299754a5aaf13ab68044102f7f42f9dc`** -- **the same 64 hex digits as the shipped frontier contributor**. The "recalibrated" artifact is not a near-copy of the shipped one; it is the same file.

**The persistence was protected explicitly, and it needed to be.** The estimator writes the lag-1 ratio as `phi` and never writes `phi_final`; the stamping script falls back to the bare `phi` when `phi_final` is absent. A refit of `rho` alone would therefore have silently stamped `copula_phi = 0.8122` in place of the frozen 1.0 -- a second, undeclared change to a separately frozen parameter. `scripts/artificial_humans/freeze_phi_in_params.py` carries `phi_final = 1.0` and PR #165's boundary ruling across verbatim, and refuses if the shipped value is anything but 1.0. This is the one trap in the procedure and a successor recalibrating either parameter will hit it.

### Step 4: the 22 rows, frontier stack (measured, gated -- Raven job 30326959, 1 min 15 s)

The simulation's `per_round.parquet` is **byte-identical** to the parent's `_simtimeout` run (sha256 `b766ca6c...` on both), which is what a byte-identical artifact under a fixed seed must produce and what the noise-off arm had already shown the machinery does. All 22 rows, the mean and the rows <= 1 count therefore come out **exactly equal**; `plots/data_analysis/evaluation/contribution_copula_recalibrated/before_after.md` carries the full table with every seed sd beside a delta of 0.0000.

| row | before | after | delta | seed sd | in seed sd | legible |
|---|---|---|---|---|---|---|
| CA | 0.8422 | 0.8422 | 0.0000 | 0.1878 | 0.00 | no |
| CB | 0.8164 | 0.8164 | 0.0000 | 0.2153 | 0.00 | no |
| CC | 0.8154 | 0.8154 | 0.0000 | 0.1333 | 0.00 | no |
| CD | 0.7665 | 0.7665 | 0.0000 | 0.1878 | 0.00 | no |
| CE | 0.9675 | 0.9675 | 0.0000 | 0.0581 | 0.00 | no |
| CF | 0.8152 | 0.8152 | 0.0000 | 0.1413 | 0.00 | no |
| **CG** (target) | **1.8449** | **1.8449** | **0.0000** | 0.3014 | **0.00** | **no** |
| SA | 0.7741 | 0.7741 | 0.0000 | 0.1618 | 0.00 | no |
| SB | 1.0695 | 1.0695 | 0.0000 | 0.0456 | 0.00 | no |
| **SC** (watch) | 1.8295 | 1.8295 | 0.0000 | 0.1355 | 0.00 | no |
| PA | 0.6408 | 0.6408 | 0.0000 | 0.0404 | 0.00 | no |
| PB | 0.8995 | 0.8995 | 0.0000 | 0.0231 | 0.00 | no |
| PC | 0.9147 | 0.9147 | 0.0000 | 0.0359 | 0.00 | no |
| PD | 0.6860 | 0.6860 | 0.0000 | 0.0593 | 0.00 | no |
| RCA | 1.7761 | 1.7761 | 0.0000 | 0.1412 | 0.00 | no |
| RCB | 1.0668 | 1.0668 | 0.0000 | 0.1425 | 0.00 | no |
| **RCC** (watch) | 1.4983 | 1.4983 | 0.0000 | 0.1631 | 0.00 | no |
| RCD | 1.3368 | 1.3368 | 0.0000 | 0.2698 | 0.00 | no |
| RCE (protected) | 0.9474 | 0.9474 | 0.0000 | 0.1063 | 0.00 | no |
| RSA | 1.1529 | 1.1529 | 0.0000 | 0.1555 | 0.00 | no |
| RPA | 0.6426 | 0.6426 | 0.0000 | 0.0175 | 0.00 | no |
| RPB | 0.7624 | 0.7624 | 0.0000 | 0.0283 | 0.00 | no |
| **mean** | **1.0393** | **1.0393** | **0.0000** | 0.0473 | 0.00 | no |
| rows <= 1 | 14 | 14 | 0 | 3.1623 | 0.00 | no |

**All 22 rows move by 0.00 seed sd and none is distinguishable from a retrain of an unchanged model** -- which here is the strongest available statement rather than the weakest, because the two runs share an artifact byte for byte and the comparison contains no noise at all, training or sampling.

### Step 6: the protected row RCE, under the amended rule (measured)

| band | human | before | after | change | pooled SE | band slope seed sd |
|---|---|---|---|---|---|---|
| 0-4 | +0.140 +- 0.018 (n 965) | +0.102 +- 0.014 (n 2045) | +0.102 +- 0.014 (n 2045) | 0.000 | 0.00 | 0.0182 |
| 5-9 | +0.104 +- 0.024 (n 929) | +0.019 +- 0.015 (n 1906) | +0.019 +- 0.015 (n 1906) | 0.000 | 0.00 | 0.0223 |
| 10-14 | -0.077 +- 0.035 (n 560) | +0.000 +- 0.022 (n 1364) | +0.000 +- 0.022 (n 1364) | 0.000 | 0.00 | 0.0263 |
| 15-19 | -0.161 +- 0.079 (n 206) | -0.085 +- 0.054 (n 474) | -0.085 +- 0.054 (n 474) | 0.000 | 0.00 | 0.0564 |

The score holds at 0.9474 in band `<= 1`, the sign pattern is `+++-` on both sides, and **no clause fires** -- nothing moved. The amended rule is applied as: the band-drop clause takes RCE's own seed sd 0.106, the sign clause is retired on 10-14 and 15-19 and takes the band's slope seed sd on 0-4 and 5-9, and the magnitude clause keeps its two §2 qualifications and additionally takes the band's slope seed sd.

**Re-checking the parent's own firing under the amended rule** (`_timeout` -> `_simtimeout`, the same machinery, run on this branch): the **sign clause no longer fires** -- it fired on the 10-14 band, where it is now retired -- but the **magnitude clause still does**. The 10-14 slope went -0.048 -> +0.000, away from the human -0.077 (qualification a holds: after is 0.077 from the human, before was 0.029), by 1.57 pooled SE (qualification b holds) and 1.83 band-slope seed sd (the amended threshold holds). So the parent's protected-row failure **survives the amendment**, on one clause of three instead of two.

### Step 5: the state-spread diagnostic (measured -- Raven job 30327026, 1 min 11 s)

`scripts/data_analysis/copula_recal_closed_loop_variance.py`, the parent's script with its arms retargeted, teacher-forces the bare trunk over each arm's own realised history.

| arm | copula | `Var(E[c \| hist])` | retention vs human | `Var(c)` | `Var(resid)` | CG ratio |
|---|---|---|---|---|---|---|
| human histories | -- | **27.933** | 1.000 | 39.916 | 11.794 | 0.8480 |
| **R (recalibrated)** | **on, rho 0.0395** | **25.275** | **0.905** | 38.789 | 12.333 | **0.8005** |
| B (parent baseline) | on, rho 0.0395 | 25.275 | 0.905 | 38.789 | 12.333 | 0.8005 |
| **C (noise off)** | **OFF** | **21.360** | **0.765** | 33.800 | 12.604 | **0.7815** |

**The prediction in the method was right, and for a stronger reason than it was offered.** The noise-off number is unchanged at **21.360** against the human 27.933 -- not because the difference happened to be small, but because arm C loads the bare trunk, a file no stamping ever writes to, and its `per_round.parquet` is byte-identical to the parent's. The prediction that a difference would appear only with the copula on is **wrong in the only way that matters**: with the copula on there is no difference either, because the refitted copula *is* the shipped copula. Arm C reproduces the parent's 21.360 / 0.7815 and arm R reproduces its 25.275 / 0.8005 to every digit printed.

The human-vs-sim gap the parent documented is therefore untouched: **the stack reaches 21.36 of the human 27.93 worth of state spread with the noise machinery off, and the copula's episode-long persistence lifts the closed loop to 25.28 while leaving the group-spread ratio at 0.8005 against the human 0.8480.**

### Step 1: what the estimator actually reads, and a prediction made before running it (measured on the tree)

Before spending a job on the refit I established what `contribution_copula_rho.py` consumes, because the hypothesis is a claim about the calibration having seen something, and a calibration can only absorb what it is shown.

**The estimator's whole input surface.** It imports `punishment_copula_rho` (the #146 estimator machinery), `aimanager.generic.data.create_torch_data` and `aimanager.generic.graph.GraphNetwork`, and reads three files: the bare trunk artifact, `experiments/2group_8agent_50ep.csv`, and the two baseline split files. It **never imports `aimanager.manager.environment` or `aimanager.simulation.linear_ah`** -- the only two source files the parent's serving fix changed (`git show --stat 26386bc`; the single occurrence of "environment.py" under `src/aimanager/generic/` is a comment). Its marginals come from a **teacher-forced forward pass over the human histories** (`teacher_forced_rows` -> `predict_independent(..., sample=False)`), never from a simulation.

**Every one of those inputs is bit-identical to what the shipped calibration saw.**

| input | shipped calibration (2026-09-16) | this branch |
|---|---|---|
| estimator script | `contribution_copula_rho.py` | unchanged since `2731b93` (2026-08-27), i.e. before the calibration |
| imported machinery | `punishment_copula_rho.py` | changed once since (`5cc0950`), inside `main()` only -- the `--stamp-rho` flag, which this script never calls |
| base trunk | sha256 `9de0d772...` (recorded in the params JSON) | sha256 `9de0d772...` (measured) |
| human data | `experiments/` | `git log -- experiments/` is empty since 2026-09-01 |
| `generic/data.py` | -- | two additive changes since (`MAX_CONTRIBUTION` / `contribution_max`, and the `MISSING_CONTRIBUTION` constant); neither touches `contribution`, `prev_contribution`, `agent_group` or `prev_punishment`, which are the keys this contributor's encoder reads (the parent's probe) |

**And the training data never contained the defect.** `parse_agent_rounds` stores a timed-out contribution as the recorded 0 (`fillna(0)`), and `prev_contribution` is `shift(contribution, default)` -- so the cell after a timeout carries **0** in training, and only round 0 carries the dataset default of 9. The imputed 9 the parent removed lived **only** on the simulation serving path (`environment.update_contribution`), which this estimator does not touch.

**The prediction, recorded before job 30325836 ran:** the refit reproduces the shipped `rho = 0.03949863621805423` exactly, because every byte it reads is the same and the estimator is deterministic under its fixed seed 38381. If that holds, the hypothesis is refuted at the only point where it is testable -- the calibration cannot have absorbed a defect it was never shown.

### Step 5a: the noise-off arm, and a determinism control nobody had measured (measured)

`23_2g8a_copula_recal_rho0` (job 30325909, 2 min 36 s) is this branch's own noise-off run: the bare, unstamped trunk in the contributor slot, everything else as the gated stack. Its `per_round.parquet` is **byte-identical** to the parent's `23_2g8a_sim_timeout_rho0`:

```
0dc13b44a1c07f05eee0c39270b52e1a5254f51e29503cc41597f8b435a26c31  23_2g8a_copula_recal_rho0/per_round.parquet
0dc13b44a1c07f05eee0c39270b52e1a5254f51e29503cc41597f8b435a26c31  23_2g8a_sim_timeout_rho0/per_round.parquet
```

Two things follow, and the second is worth more than the first. **The noise-off arm is untouched by a recalibration**, as it must be -- it loads the bare trunk, which no stamping ever writes to, so `Var(E[c | history]) = 21.360` and the group-spread ratio 0.7815 carry over from the parent unchanged. And **the simulation is bit-reproducible across remote dirs, nodes and sessions**: the parent asserted "the runs are deterministic given the artifacts and the seed" (its note 4) without measuring it; this run measures it, on a different GPU node in a different isolated remote dir, at the level of the recorded output rather than of the scores.

### Step 4a: the evaluation environment reproduces the parent's 22 rows (control, measured)

Before scoring anything new, `python -m aimanager evaluate` (with `PYTHONPATH=<worktree>/src`, §Hygiene) was re-run over the **parent's own** `_simtimeout` simulation. It rewrote `evaluation/scores.csv`, `evaluation/metrics.csv` and all 25 figures, and `git status` came back **clean** -- every byte identical to what the parent committed. The scoring is deterministic under its master seed 42 and this worktree's environment reproduces it, so any difference the recalibration produces downstream is the recalibration's, not the scorer's.

## 4. Notes

1. **Step 1 was done before the refit, and it turns the experiment into a sharp test rather than a fishing trip.** The hypothesis in the parent's successor note is that the frontier's copula "was calibrated in the presence of the defect". Reading the estimator's input surface says it was not: `rho` is fitted teacher-forced against human histories, where a timed-out player's contribution and its lag are the recorded 0, and the imputed 9 exists only inside `environment.update_contribution` at simulation time. The refit is therefore a prediction with two possible outcomes, both informative: an unchanged `rho` refutes the absorption story, and a changed one would mean something about the estimator's inputs moved that step 1 missed.

2. **The prediction held exactly, and the whole chain was run anyway rather than argued.** `rho` came back at 0.03949863621805423 against the shipped 0.03949863621805423, all 36 params fields equal, the stamped artifact equal to the shipped one in all 64 hex digits of its sha256, the simulation's `per_round.parquet` equal to the parent's, all 22 rows equal. Four independent equalities, each measured. The argument alone would have been sound; the measurements make it checkable by someone who does not trust the argument, which is the point of running them.

3. **The hypothesis is refuted, and the correct reading of the parent's CG loss is different.** The mis-served nines were never in the calibration -- the calibration is a human-data fit and the human data records a timeout as 0. They were an extra source of between-group dispersion **in the closed loop**, sitting on top of a correctly fitted copula. So the parent's CG regression is not calibration debt being repaid; it is the removal of a spurious dispersion source that had been flattering the row. The frontier stack's CG deficit is real and was partly hidden by the defect. That also explains the asymmetry the parent found most informative: the reference stack has no copula, so its contributor had nothing compounding the spurious dispersion, and removing it simply made that stack more correct.

4. **There is no legal way to recover CG by moving `rho`, and this is the useful negative result.** The only `rho` the estimator supports is the one it measures, and it measures the shipped value. Choosing a larger `rho` because CG would score better is tuning a parameter at a metric's definition rather than at behaviour -- §5's second bullet of illegal changes -- and the §2 pre-flight is explicit that `rho` is never tuned to the group-spread ratio. The copula's *strength* is therefore closed as a route to CG on this stack. What remains open is the copula's *shape* and the trunk's state.

5. **The persistence trap.** See step 3. `contribution_copula_rho.py` writes `phi` (the lag-1 ratio) and never `phi_final`; `make_contribution_copula_artifact.py` stamps `phi_final` when present and the bare `phi` otherwise. A refit that only means to move `rho` will move `phi` from 1.0 to 0.8122 unless `phi_final` is carried over. Nothing warns you: both values pass every assertion in the stamping script. `scripts/artificial_humans/freeze_phi_in_params.py` now does the carry-over and refuses to guess.

6. **Two controls fell out of this run that the campaign did not have, and both are worth keeping.** (a) **The simulation is bit-reproducible**: the noise-off arm, re-run from a different isolated remote dir on a different GPU node, produced a `per_round.parquet` byte-identical to the parent's, and so did the frontier arm. The parent asserted determinism; it is now measured at the level of the recorded output. (b) **The evaluation environment is bit-reproducible**: re-running `python -m aimanager evaluate` over the parent's own simulation rewrote `scores.csv`, `metrics.csv` and all 25 figures and left `git status` clean. Together these mean a future before/after on unchanged artifacts carries **zero** run-to-run noise, and any non-zero delta in such a comparison is a real code difference, not a draw.

7. **The cluster's CPU partition was unusable for the whole session** (`QOSGrpCpuLimit` on `mpib_cpu`, every CPU job pending at 0:00 indefinitely). The refit, the stamping and the teacher-force job were moved to the `gpu` partition, where the account had capacity; all three are pure CPU work and the GPU sits idle in each. The SLURM files record this. It changes nothing about the numbers -- the refit reproduced a value computed on the `small` partition to the last digit, which is incidentally also a check that the estimator is not sensitive to the node it lands on.

8. **Housekeeping.** Nothing was retrained; every slot in every run loads the same file on disk as the parent's runs. The punisher's severity copula (`rho` 0.4273) was not touched -- it is a different model family's frozen parameter. Scratch files live under `/private/tmp/.../scratchpad`. The remote dir `~/repros/ai-runs/copula-recal` can be deleted when this PR closes. `src/` is untouched, `flake8` is clean over it, and `black --check` reports only the two files already unformatted on the parent (`artificial_humans/train.py`, `rl_manager.py`), which were left alone.

## 5. The verdict: [FAIL] on the gates, and a refutation of the hypothesis

### The hypothesis is refuted

**The old calibration had not absorbed the defect, and could not have.** The contribution copula's `rho` is estimated by pairwise-likelihood MLE against **human** histories, teacher-forced, from a training tensor in which a timed-out player's contribution and the lag that follows it are the recorded **0**. The imputed 9 the parent removed existed only in `environment.update_contribution`, on the simulation serving path, which this estimator does not import and never executes. Refitting against the corrected tree returns `rho = 0.03949863621805423` -- **the shipped value to the last digit, a move of exactly 0.0 against a bootstrap SE of 0.0098 and a 95% CI of [0.0181, 0.0566]**. That is inside the estimator's ordinary spread in the only sense available: it is not a movement at all.

The parent's reading of its own CG loss should be replaced. The mis-served nines were a spurious source of between-group dispersion **in the closed loop**, not a bias the calibration had internalised. Removing them did not create calibration debt; it exposed a CG deficit the defect had been covering. A refutation here is worth as much as a confirmation would have been, and the serving fix stands on its own measurement either way: it is still correct, it still moves `Var(E[c | history])` from 18.92 to 21.36 against the human 27.93 with the noise machinery off, and it still improves the reference stack.

### Gates: [FAIL]

**Gate 1 -- a band upgrade on the declared target CG: FAIL.** CG is 1.8449 before and 1.8449 after; the move is 0.0000, i.e. 0.00 of its 0.3014 seed sd, and there is no band change to clear a floor with. (Noted for the record: §2 as amended lists CG among the ten rows that cannot serve as a gate-1 target on a single run at all. The gate was declared on CG regardless, per the experiment's brief; it fails on the movement, not on the technicality.)

**Gate 2 -- the 22-row mean: PASS.** 1.0393 -> 1.0393 against a ceiling of 1.1432. `rows <= 1` holds at 14/22.

**The protected row RCE: does not fire.** Score 0.9474 on both sides in band `<= 1`, all four band slopes unchanged to three decimals, no clause of the amended rule triggered.

**Watch rows.** SC 1.8295 -> 1.8295 and RCC 1.4983 -> 1.4983, both exactly unchanged. Neither of the two rows the parent lost legibly is recovered by a recalibration, because there is no recalibration to recover them with.

### Verdict tag

**[FAIL]** -- no band upgrade on the declared target, because the declared change turned out to be the null change. Gate 2 passes and the protected row is untouched. The experiment's value is the refutation and the four equalities that establish it, not the score line.

## 6. For a successor

1. **Do not re-open the copula's strength on this stack.** `rho` is the value the estimator measures on human data, the estimator is invariant to the serving path, and the refit is reproducible to the last digit. Moving `rho` away from 0.0395 to buy CG would be tuning at the metric rather than at behaviour (§5, illegal), and the pre-flight in the calibration script says in as many words that `rho` is never tuned to the group-spread ratio. **The strength is closed; the shape is not.**

2. **The open CG route is the copula's shape and the trunk's state, and both have measurements waiting.** With the noise machinery off the stack reaches `Var(E[c | history])` = 21.36 against the human 27.93, and the residual variance is already right (12.60 against 11.79) -- the gap is in what the model conditions on, not in how much noise it adds. Separately, `copula-missing-state.md` measured the human residual dependence to be a **round-local shock with a one-round echo**, not the episode-long latent that is shipped; §2 keeps the shipped shape only because nothing yet replaces the variance it supplies. A copula with the human's shape, calibrated by the same estimator, is a declarable experiment with a target of CG and an argument behind it that this branch's result does not touch.

3. **A `phi` refit is still unasked and is now cheap to ask.** `phi` is stamped at the unit-root boundary 1.0 by PR #165's ruling, while the estimator's own lag-1 figure is 0.8122 with a CI of [0.316, 1.840] that straddles 1. It is a separate frozen parameter and a separate declared experiment; the machinery for it is on this branch (`freeze_phi_in_params.py` inverted, i.e. stamping `phi_final` from the estimate), and the diagnostic to judge it with is the per-round decomposition in `copula_recal_closed_loop_variance.py`, where an episode-long latent and a decaying one separate cleanly.

4. **The parent's protected-row firing survives the amendment on one clause.** Its sign clause fired on the 10-14 band and is retired there; its magnitude clause still fires, at 1.57 pooled SE and 1.83 band-slope seed sd, with the slope moving away from the human. Anyone re-reading PR-level verdicts under the amended rule should re-run the check rather than assume retirement clears them -- `scripts/data_analysis/copula_recal_table.py::amend` implements the amended rule and takes any before/after pair.

5. **Before any future before/after on unchanged artifacts, know that the floor is exactly zero.** Note 6: both the simulation and the evaluation are bit-reproducible across nodes, remote dirs and sessions. A comparison that changes only an artifact's metadata will show either all-zeros or a genuine code difference; there is no third possibility and no draw to blame.

6. **The evaluation suite still scores a simulated timeout and drops a human one** -- the parent's successor note 2, unchanged by this branch and still needing the maintainer, because `evaluation_suite/` is frozen surface. It is the last place in the pipeline where the imputed 9 does any work.
