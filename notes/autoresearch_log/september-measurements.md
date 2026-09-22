# The September 2026 measurements: index and consolidation record

Six measurement-only pull requests -- #183, #186, #187, #188, #191 and #195 -- are consolidated here. None of them changes a model's behaviour: each is a number plus the tooling that produced it. Their six logs are carried unchanged; this file indexes them, and records what was left behind on the source branches so that a reader knows the evidence exists even though it is not here.

The consolidation is a re-selection, not a rewrite. Every analysis script, every table another artefact reads, and every trained artifact that the noise-floor rule now requires for re-measurement is here at its original path. Bulk simulation outputs, per-run figures and intermediate teacher-forced frames are not; they are regenerable from the configs and artifacts that are.

## The six

### PR #183 -- the punishment response is learned, not memorised

`rcb-holdout-teacher-forced`, log: [`rcb-holdout-teacher-forced.md`](rcb-holdout-teacher-forced.md)

The RCB row's teacher-forced statistic of 0.09 on the best contributor trunk was an in-sample number, so it could have been memorisation rather than a learned response. Retraining the trunk five times with one cross-validation fold held out and teacher-forcing each model on the games it never saw gives a pooled held-out statistic of **0.0951** against **0.0823** in sample and **0.797** in self-play. The response is present out of sample and lost only in the closed loop, which rules generalisation fixes out and leaves closed-loop drift as the family that can move the row. Two secondary deficits survive the test: the 15-19 band's withdrawal is about 40% as strong out of sample as in, and the 10-14 band has the wrong sign in every condition.

### PR #186 -- the shared draw refills collapsed variety, it does not restore a correlation

`auto/copula-closed-loop-variance`, log: [`copula-closed-loop-variance.md`](copula-closed-loop-variance.md)

Three closed-loop arms of one stack, identical but for the contribution copula's fields, separate the two readings of what the copula does. Without it the loop keeps its per-round noise (predictive SD 3.09 in every arm) and loses the spread of the states the players reach: **Var(E[c | history]) 18.9 against 23.9 with the copula and 27.9 on human histories**. Turning the correlation on but stripping its persistence reproduces the human within-round residual correlation almost exactly (**0.031 against 0.032**) and buys only about a fifth of the CG gap; the persistence carries the rest, compounding a 0.49-point per-round push into a 2.57-point shift in the group's level.

### PR #187 -- a seventh of the residual correlation is observable, and none of it is static

`auto/copula-missing-state`, log: [`copula-missing-state.md`](copula-missing-state.md)

On the human games the trunk already removes 93% of the raw within-group co-movement, leaving rho = 0.0479. Group state the trunk does not see explains **14% of that [9%, 29%]** -- the share of the group punished last round, the group's spread, and its trend -- and at most 23% with every legal candidate at once. The remainder is not a slowly varying group effect: the cross-player correlation of the residual latent is 0.036 within a round, 0.024 one round later and **-0.005 [-0.010, +0.000] pooled over every pair of rounds two or more apart**. The stamped static latent has no counterpart in the human residuals; what they support is a round shock with a two-thirds echo.

### PR #188 -- model uncertainty is a sixth of the dose and has the wrong timescale

`auto/copula-seed-ensemble`, log: [`copula-seed-ensemble.md`](copula-seed-ensemble.md)

Five retrainings of the shipped contributor, differing only in seed, measure how much the model's own error would move if the weights were drawn rather than fixed. The members disagree by 0.38 contribution points per agent-round, 2% of one model's predictive variance; the disagreement is genuinely shared within a group (within-cell correlation 0.16 to 0.24) but translates to a copula-equivalent **rho of 0.005 to 0.009 against the fitted 0.0395, whose interval starts at 0.018**. It also decays with a half-life of about a round (lag-1 0.53, ICC 0.14) where the shipped latent is held static for 24. Drawing one member per episode in the closed loop scores like no copula at all (CG 2.68 against the copula's 1.55).

### PR #191 -- a location-scale head extrapolates worse, not better

`auto/head-state-spread-diagnostic`, log: [`head-state-spread-diagnostic.md`](head-state-spread-diagnostic.md)

The argument for building an ordinal location-scale emission was that 21 free logits would relax toward the training marginal off the human manifold. They do not. With the copula off, the Gaussian heads retain **Var(E[c | history]) of 16.5 and 13.3 against the categorical head's 18.9**; off the manifold their gain is 0.90-0.96 and 0.85-0.92 against 0.95-1.02, and the categorical head is the only one whose gain rises rather than sags at the extremes. Matching the switch model closes the one confound and moves the categorical number by +0.006. Two components survive the result: the inflation at the corners and at repeat-previous, which is what makes the inflated head's noise calibrated; and the persistent latent, worth about twice what any head choice is worth.

### PR #195 -- the noise floor of the evaluation

`auto/seed-spread-noise-floor`, log: [`seed-spread-noise-floor.md`](seed-spread-noise-floor.md)

Six contributors of one architecture -- PR #188's five members plus the shipped artifact -- run through one stack at one simulation seed give an error bar for every row of the 22-row evaluation. **A typical row moves by sd 0.138 and spans 0.341 end to end**; the 22-row mean moves by sd 0.047 against a gate-2 allowance of 0.103; the rows-at-ceiling count moves by sd 3.16, from 6 to 14. **Ten of the 22 rows cannot be judged on a single run**, the protected row RCE among them, because a band boundary lies inside one seed sd and the six arms straddle it. PR #194's disputed level shift is reproduced by a plain reseed (10.327 against the candidate's 10.355), and the shipped baseline the recent verdicts were measured against is first of six on every aggregate.

## What the consolidation keeps

- **The six logs**, byte-identical to their source branches. Where a log names a file that was not carried, that file is on its source branch at the same path.
- **Every analysis and tooling script** from all six, at its original path and with its behaviour unchanged. Formatting was normalised (black, and targeted `noqa` markers for long report strings); the parsed syntax tree of every script is identical to the source branch's.
- **Every table that something else reads.** Both atlas generators and `notes/autoresearch.md` cite files in these six by path; all 29 of those paths are here.
- **PR #188's five seed contributors** and the ensemble manifest, the material the noise floor was measured from and what would let it be re-measured on another stack.
- **The configs that define every measured arm**, so each simulation and each training can be re-run rather than only read about.
- **PR #188's ensemble sampler** (`src/aimanager/simulation/ensemble_ah.py`) and its one-line dispatch in `load_ah_model`.

## What it drops, and where it still is

Everything below stays on its source branch at the path the logs give.

| dropped | where | why |
|---|---|---|
| Bulk simulation output: `per_round.parquet`, `aggregates.csv`, per-run figures and the whole `evaluation/visuals/` trees of the arms | #186, #188, #191, #195 | regenerable from the committed configs and artifacts; the one exception, the seed-ensemble run's `evaluation/scores.csv`, is cited by a report and is here |
| Intermediate teacher-forced frames: the `tf_*.parquet` arms (#186, #191), the residual table (#187) and the two `*_per_row.parquet` frames (#188) | #186, #187, #188, #191 | inputs to the committed summary tables, reproduced by the scripts that are here |
| PR #183's ten held-out fold artifacts | `rcb-holdout-teacher-forced` | see below |
| PR #188's per-member metrics and confusion parquets (ten files) | `auto/copula-seed-ensemble` | training evidence for a closed question; the cross-validated log losses they carry are quoted in both #188's and #195's logs |
| Derived artifacts: the phi = 0 copula stamp (#186), the two rho-zero Gaussian bundles (#191), the five copula-carried seed stamps (#195) | #186, #191, #195 | each is a one-transform copy of a parent, and the script that makes it is here (`stamp_copula_phi0.py`, `stamp_contribution_copula_rho0.py`, `carry_contribution_copula_params.py`) |
| PR #183's three secondary summary CSVs (the local-CPU cross-check and the stimulus-skip trunk) | `rcb-holdout-teacher-forced` | their tables are printed in full in the log |
| Per-run figures and the markdown twins of committed CSVs (`per_row.md`, `tables.md`, `scores_22.md`, `*.jpg`) | all six | the numbers are in the CSVs and in the logs |

**The fold artifacts.** PR #183's ten held-out models were dropped, and PR #188's five seed models were kept, on the distinction the rules draw. The seed members are the material a noise floor is measured *from*: the rule that a declared target must clear the floor means a successor re-measures it on its own stack, and that needs these weights. The fold models proved a question that is now settled -- the response is learned -- and nothing downstream re-measures anything from them; both fold configs are here, and a fold is 2m25s on one GPU. They are also inert on `main`, which carries neither the per-group virtual node nor the stimulus skip they were trained with.

## Notes on what the consolidation could not carry

1. **PR #186's latent logging is not here.** It touched three files. Two of them apply to `main` (`manager/environment.py`, `simulation/simulate.py`); the third, `generic/graph.py`, does not -- `main` has no contribution copula anywhere under `src/`, so the method the logging hooks into does not exist. Carrying only the two that apply is not behaviour-neutral: the state entry and the guarded merge would add an eleventh, all-`NaN` `copula_z` column to every `per_round.parquet` `main` produces, and nothing would ever fill it. Under the rule that an addition which changes an existing path stays out, it stays out. It belongs with the copula lineage, and lands when that lineage lands. `scripts/data_analysis/copula_closed_loop_variance.py`, which consumes the column, is here.
2. **PR #188's sampler is here and is purely additive.** `ensemble_ah.py` is a new module, imported lazily and only from the new branch of `load_ah_model`, which fires on a `.ensemble.yml` suffix that no config on `main` uses. Every other path through `load_ah_model` is byte-identical: the `.joblib` branch returns before it, and the `GraphNetwork` fall-through after it is untouched. Like the artifacts, it is inert until the copula lineage merges -- it asserts `copula_rho == 0` on its members, and `main`'s `GraphNetwork` has no such attribute.
3. **The scripts run against the copula lineage, not against bare `main`.** They import `contribution_copula_rho`, `punishment_copula_rho`, `rcb_teacher_forced`, `gmlp_group_copula_diagnostic` and `stamp_contribution_group_copula`, which are committed on the unmerged experiment branches, and a `GraphNetwork` that knows about copulas, virtual nodes and the stimulus skip. They are carried so that the measurements can be reproduced and extended once that lineage is on `main`, not so that they run on it today.
4. **Both reports now read from `main`.** `scripts/reports/build_rebaseline_atlas.py` and `build_punisher_rebaseline_atlas.py` used to pull the numbers of these six straight off six experiment branches, and the held-out teacher-forced table out of a scratch worktree. They now read all of it from one branch, named by `SEPT` at the top of each file and defaulting to `main`. Both rebuild byte-identically after the change. The punisher report's file cache had to start keying on the full path rather than the basename, because one branch now ships `round_blocks.csv` under two analysis directories.
