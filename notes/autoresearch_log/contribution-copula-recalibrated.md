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
| 2 | Refit `rho` with the existing estimator (`contribution_copula_rho.py`, same flags, same seed) against the corrected tree; report the estimate, its bootstrap CI and SE, and whether the move from 0.0395 is inside the estimator's ordinary spread. | pending |
| 3 | Stamp the refitted value onto a copy of the contributor artifact and verify the copy is weight-identical to its base outside the three copula fields (dict-level bit comparison plus the teacher-forced probability check the stamping script already runs). | pending |
| 4 | Simulate the frontier stack with the recalibrated contributor and evaluate all 22 rows against the parent's, every movement quoted beside its PR #195 seed sd. | pending |
| 5 | Run the noise-off arm and the state-spread diagnostic; record what the recalibration does and does not change about `Var(E[c | history])` and the group-spread ratio. | pending |
| 6 | Check the protected row RCE under the amended rule (band-drop threshold 0.106, sign clause retired on 10-14 and 15-19), reporting every band slope with its standard error and row count. | pending |
| 7 | Judge, log, PR against `auto/sim-timeout-imputation`. | pending |

## 3. Results

*(the results table is filled in when the simulation lands)*

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

## 4. Notes

1. **Step 1 was done before the refit, and it turns the experiment into a sharp test rather than a fishing trip.** The hypothesis in the parent's successor note is that the frontier's copula "was calibrated in the presence of the defect". Reading the estimator's input surface says it was not: `rho` is fitted teacher-forced against human histories, where a timed-out player's contribution and its lag are the recorded 0, and the imputed 9 exists only inside `environment.update_contribution` at simulation time. The refit is therefore a prediction with two possible outcomes, both informative: an unchanged `rho` refutes the absorption story, and a changed one would mean something about the estimator's inputs moved that step 1 missed.
