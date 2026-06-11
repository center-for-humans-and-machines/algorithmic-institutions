# [ACTIVE] Fix 2-group RL manager training (Issue #103)

Tracking issue: [#103](https://github.com/cemrtkn/algorithmic-institutions/issues/103). Follow-on to the 100k 1-group tuning landed in [#102](https://github.com/cemrtkn/algorithmic-institutions/pull/102) and the 2g8a infrastructure landed in [#69 plan](doc/plans/69-two-manager-rl-training.md).

## Goal

Move the 2-group, 8-agent, group-switching RL training from the initial 4k smoke runs (`03_2g8a_{sum,avg}.yml`, `n_update_steps: 4000`) to a properly tuned, properly horizon-sized run, so we can answer the open question from #92: is the `mean_punishment -> 0` collapse a genuine policy optimum against this AH stack, or an artefact of premature stopping / under-tuned hyperparameters? The answer determines whether the next bottleneck is the manager training loop or the AH stack itself (#84, #92).

### Rescope note: RNN-flush fix is already on main

The framing in #103 leans on porting #102's RNN-flush fix to the 2-group path. That port is already done on `main`:

- `src/aimanager/rl_manager.py:68-70` — RL manager: `manager.get_action(state, first=round_number == 0, greedy=on_policy)`.
- `src/aimanager/rl_manager.py:80-84` — opponent manager: `opponent_manager.predict(state, reset_rnn=round_number == 0, ...)`.

So there is **no code-level RNN-flush change to make**. The remaining axes from #102 that genuinely transfer to 2-group are (a) `lr` and `target_update_freq` tuning. New axes specific to #103 are (b) runtime estimation against the Raven 24h cap (which sets the horizon), (c) a quick audit of existing checkpoint/resume support (designing/implementing it is out of scope -- if needed it gets its own issue), and (d) an explicit AH-stack tripwire so we don't burn cluster time chasing a saturated AH response curve.

## Plan

| # | Section | Change |
|---|---------|--------|
| 1 | Runtime estimation + resume-support audit | Inspect the existing 4k-step `03_2g8a_sum` run on Raven (`.log/training/manager/rl_manager/03_2g8a_sum/fce64a11/`) for seconds/step; derive `max_steps_per_job`. Also: read the codebase for any existing checkpoint/resume support, document the finding. |
| 2 | 4-cell factorial training | Add four sum-reward configs (`03_2g8a_sum_a` … `_d`) varying `lr` and `target_update_freq`, each at `n_update_steps = max_steps_per_job`. |
| 3 | AH-stack tripwire | Explicit numeric criterion to halt training and escalate to supervisors instead of further tuning. |
| 4 | Comparison sim + plots | `15_2g8a_factorial.yml` simulation evaluating all 4 cells; metrics + comparison plots. Sim outcome drives the "push beyond a single job?" decision -- handled in a separate issue if needed. |

### 1. Runtime estimation + resume-support audit

**1a. Runtime estimation -- reuse existing 4k-step run**

- **What:** No new smoke job needed. The existing `03_2g8a_sum` 4k-step training on Raven at `.log/training/manager/rl_manager/03_2g8a_sum/fce64a11/` already gives us the unit cost on the exact stack we care about. Concretely:
  1. On Raven, inspect the SLURM log under `.log/training/manager/rl_manager/03_2g8a_sum/fce64a11/` -- look for the tqdm bar's steady-state iterations/sec (skip the first ~50 warm-up steps to discount torch lazy-init overhead).
  2. Run `seff <jobid>` against the SLURM job id from that run for an authoritative wall-time + CPU/GPU efficiency summary.
  3. Cross-check: total wall time / 4000 should match the steady-state iter/sec from step 1 (within a few %); discrepancy means the warm-up tail or eval-period overhead is non-negligible and should be accounted for in the projection.
- **Why this is sufficient:** the 4k run used the same config (`03_2g8a_sum.yml`), same AH stack, same opponent, same batch_size, same `switch_every`, same `eval_period` -- only differing from §2's runs in `n_update_steps` and (downstream) `target_update_freq`/`lr`. Those don't change per-step cost. Submitting a fresh 1k-step smoke would just re-measure the same number.
- **Derived value:** `max_steps_per_job = floor(training_budget_seconds / seconds_per_step)`. Use a `training_budget` of ~22h (24h Raven cap minus margin for job-setup, AH loading, final save + parquet write, eval-period overhead). This number is the input to §2's `n_update_steps`.

**1b. Resume-support audit**

- **What:** read `src/aimanager/rl_manager.py` (train loop, save/load paths) and `src/aimanager/manager/manager.py` `save`/`load` to confirm what's already supported. The expected answer is "only final-artefact save of `policy_model`; no mid-training checkpoint, no optimizer/replay/target-model persistence". Document the finding in this plan or on #103.
- **Why:** #103 explicitly asks "Is checkpointing already supported? can one just keep training a model saved in a previous training run?". Answering it is in scope; *designing or implementing* checkpoint/resume is **not** in scope for this issue. If we end up wanting a horizon longer than `max_steps_per_job` (see §4), we open a separate issue for checkpoint/resume work.

**Deliverable:** a one-line addendum at the bottom of this plan file (or a comment on #103) recording (i) `seconds/step`, (ii) derived `max_steps_per_job`, (iii) one-sentence summary of existing resume support. The engineer doing §2 should not start until (i)+(ii) are filled in.

### 2. 4-cell factorial training

This is the experiment of #103. Run four trainings on the `sum` reward-mode, each at `n_update_steps = max_steps_per_job` from §1, varying `lr` and `target_update_freq`. The cells are picked to (a) replicate the legacy baseline, (b) port #102's winning recipe, (c) test whether a yet-higher `lr` helps now that we have evidence `5e-4 >> 2e-4` on 1-group, and (d) check whether a less aggressive sync rate at high `lr` is more stable.

| Cell | Config file | `lr` | `target_update_freq` | Purpose |
|------|-------------|------|----------------------|---------|
| A | `03_2g8a_sum_a_legacy.yml` | 2.e-4 | 1000 | Legacy baseline -- matches current `03_2g8a_sum.yml` at the new horizon. |
| B | `03_2g8a_sum_b_lr5e4.yml` | 5.e-4 | 200 | Port of #102's "all 3 fixes" 100k recipe. |
| C | `03_2g8a_sum_c_lr1e3.yml` | 1.e-3 | 200 | Same freq as B; isolates the effect of an even higher `lr`. |
| D | `03_2g8a_sum_d_lr1e3_freq500.yml` | 1.e-3 | 500 | High `lr` paired with a less aggressive sync rate; checks whether high-lr training stabilises with rarer target updates. |

- **Where:** `configs/training/rl_manager/`. For each cell, start from the current `03_2g8a_sum.yml` and change only `manager_args.opt_args.lr`, `manager_args.target_update_freq`, `n_update_steps`, `job_id`, and `output_dir` (e.g. `artifacts/manager/03_2g8a_sum_a_legacy`). Leave `replay_memory_args`, `env_args`, `eval_period: 20`, `training_batch_size: 1`, `manager_args.model_args`, and all AH/opponent/switch paths untouched.
- **Reward mode:** sum only. `avg` was already validated parity-wise in #92 and is not the headline reward formula for the 2-group setup.
- **Why these axes, not a full factorial:** #102 already established that `lr=5e-4` beats `lr=2e-4`. We don't re-test that here -- we extend the `lr` sweep upward (1e-3) and the `freq` sweep (200 vs 500 at high `lr`). Cell A is the legacy anchor so we can quote a comparable "before" curve.
- **Sequencing:** this section is blocked on §1a's measurement. Don't write these configs until `max_steps_per_job` is known.

### 3. AH-stack tripwire

- **What:** during the tuned training run, monitor wandb's `eval/punishment` curve. Halt and escalate to supervisors (do not further iterate on hyperparameters) when **all of**:
  - By update_step 20000, `eval/punishment < 0.5` (i.e. the manager has effectively stopped punishing -- the #92 finding).
  - A post-hoc counterfactual probe using #89's `intervention_probe.py` against the trained checkpoint shows `|Δcontribution| < 0.5` per `+5` punishment dose on the contribution AH used in `03_2g8a_*` (`artifacts/artificial_humans/group_switching_contribution_50ep/model/...`).
  - The `eval/rl_group_size` curve (logged via `src/aimanager/rl_manager.py:333-340`) shows < 1.0 net change vs the initial 4 agents -- i.e. the switch dynamics aren't compensating either.
- **Where:** monitoring is a notebook / hand check, no code change. The criterion lives in this plan; whoever runs the tuned training quotes it in the PR / issue comment.
- **Why:** #84 already showed the 2g8a contribution AH has near-zero sensitivity to `prev_punishment` (shuffle importance +0.004 vs +0.14 in the legacy 1-group AH). If the manager + switch dynamics can't extract reward from punishment either, the binding constraint is the AH response curve, not the manager training loop. Burning cluster hours on further `lr`/`target_update_freq` sweeps is wasted effort in that regime.

### 4. Comparison simulation + plots

- **Simulation config:** `configs/simulation/manager_testing/15_2g8a_factorial.yml`. Mirror the structure of `configs/simulation/manager_testing/14_100k_three_way.yml` (the #102 three-way comparison) but with the four cell checkpoints (`03_2g8a_sum_{a,b,c,d}`) as the managers under test. Same AH stack, same opponent AH, same switch predictor as training -- in-distribution evaluation.
- **Validation filter:** before plotting, drop any cell whose training metrics show it never escaped the `mean_punishment ≈ 0` regime *or* failed to improve `group_payoff_sum` over the run. Only cells with "healthy-looking" training metrics get sim-evaluated -- this is the explicit validation step before the horizon-extension decision.
- **Plots:**
  - `plots/rl_manager/03_2g8a_factorial/metrics.png` -- per-cell training-time metrics via `scripts/plotting/plot_rl_manager_metrics.py` over the four parquets, side-by-side like #102's `02_20k_factorial/metrics.png`.
  - `plots/simulation/15_2g8a_factorial/comparison_manager.jpg` -- side-by-side per-round trajectories for contribution, punishment, group_payoff, group_payoff_sum, rl_group_size.
- **Decision after sim:** if at least one cell produces an adequate policy in simulation, ship the best of the 4 as the new `03_2g8a_sum` reference and close #103. If sim shows all four are inadequate (e.g. nothing meaningfully beats the legacy `03_2g8a_sum` checkpoint on `group_payoff_sum`), the working hypothesis becomes "the single-job horizon is too short" -- but we **do not implement checkpoint/resume in this issue**. Open a new issue for that work and route the next step through it. (See "Out of scope" below.)
- **Why:** PR-evidence parity with #102. Embedded plots in the PR body let reviewers eyeball which cell, if any, moved the needle on 2g8a; this is also the input to the §3 tripwire (does the manager learn beyond `p≈0` and improve sum payoff?).

## Implementation notes

- For the runtime estimate (§1a), the SLURM job id for `fce64a11` should be in the log dir's filenames (`slurm-<id>.out`) or the job's metadata; `seff <jobid>` gives one-line authoritative wall-time, CPU, and memory efficiency. Use it alongside the steady-state tqdm rate.
- For the resume-support audit (§1b), the relevant code is concentrated in `src/aimanager/rl_manager.py` (`train_manager` body around the `manager.save(model_file)` call near line 342) and `src/aimanager/manager/manager.py` `save`/`load`. The audit is read-only -- it produces a one-sentence finding, not a code change.
- All four §2 cells share the same AH stack, opponent AH, switch predictor, and env args. Only `manager_args.opt_args.lr`, `manager_args.target_update_freq`, `n_update_steps`, `job_id`, and `output_dir` differ between them. Keep that scope discipline so the comparison plot in §4 reflects only the variables of interest.

## Out of scope

- **Designing or implementing checkpoint/resume.** §1b only audits whether it is *already* supported. If §4's sim-validation step shows we need a horizon longer than a single 24h job, that work goes into a **new** issue -- #103 ships the tuned-hyperparameter result regardless.
- **SLURM `SIGUSR1` graceful-checkpoint handler.** Same reason -- moves with the checkpoint/resume work if/when that issue is opened.
- AH stack retraining or re-tuning (contribution AH, switch predictor, autoreg opponent). The tripwire in §3 routes this to supervisors.
- Replacing the autoregressive punishment opponent with a non-autoregressive variant.
- Self-play (both groups RL-controlled).
- More than 2 groups, or asymmetric group sizes.
- A wider factorial in `lr` × `target_update_freq` -- the 4 cells in §2 are the agreed scope.

## Next Actions

- [x] §1a: inspect the existing 4k-step run at `.log/training/manager/rl_manager/03_2g8a_sum/fce64a11/` on Raven; record seconds/step and derived `max_steps_per_job` in this plan / on #103. **Result below.**
- [x] §1b: read `rl_manager.py` + `manager.py` save/load; record a one-sentence finding on whether resume is already supported. **Result below.**
- [x] §2: add the four cell configs (`03_2g8a_sum_a_legacy.yml`, `..._b_lr5e4.yml`, `..._c_lr1e3.yml`, `..._d_lr1e3_freq500.yml`) with `n_update_steps = 15000` from §1a.
- [x] §2 launch: submit the four trainings on Raven. Submitted 2026-05-28; SLURM ids A=27529563 (uuid 02679843), B=27529565 (9258a977), C=27529566 (49e4b22f), D=27529574 (bdf446b4); all 20h time limit, pending at submission.
- [ ] §3 monitoring: at update_step 20000 of each healthy cell, evaluate the tripwire; escalate to supervisors or continue.
- [ ] §4: add `15_2g8a_factorial.yml` simulation; produce training-metrics and simulation-comparison plots over the four cells.
- [ ] If sim shows all four cells inadequate: open a new issue for checkpoint/resume + longer-horizon training; do not extend #103's scope.
- [ ] Open PR with the four configs + plots embedded; cross-link #103.

## §1 Findings (recorded 2026-05-28)

**§1a -- Runtime estimate from existing `fce64a11` run** (`.log/training/manager/rl_manager/03_2g8a_sum/fce64a11/`):

- Job: SLURM id `27159387`, 4000 update steps on `gpu:a100:1`, 4 CPUs, 16 GB RAM.
- Wall time: `05:33:08` (19988 s) per `sacct`.
- Steady-state tqdm rate (end of run): **3.99 s/it**.
- Wall-time amortized: 19988 s / 4000 steps = **4.997 s/step**. The extra ~1 s/step over the steady-state rate is the warm-up tail (first ~20 steps were 4-8 s/it -- torch lazy-init, CUDA kernel compile), eval-period rollouts every 20 steps, and the final save + parquet write. This overhead will recur in every run, so the **wall-time figure is the right unit for projecting `max_steps_per_job`**.
- Derived `max_steps_per_job` at 22h budget: `floor(22 * 3600 / 4.997) ≈ 15850 steps` -> use **`n_update_steps = 15000`** for the four §2 cells (round down for a safety margin against eval/save tail and any per-job variability).
- `target_update_freq` from #102's "~500 syncs" rule: `15000 / 500 = 30` -> would round to **50** as the floor; but cells B/C/D have explicit freqs (200, 200, 500) per the agreed factorial, so the rule only applies to cell A (legacy keeps 1000).

**Implications:** a single Raven job at the legacy config reaches ~15k steps, comfortably above #92's 4k smoke but well short of #102's 100k. The four §2 cells fit in single jobs each. If the §4 sim verdict is "inadequate", checkpoint/resume + longer-horizon training is the natural next step -- separate issue per the agreed scope.

**§1b -- Resume support audit:**

- `manager.save` (`src/aimanager/manager/manager.py:193-201`) persists **only** `policy_model` + `n_contributions`, `n_punishments`, `n_groups`, `default_values`. No `target_model`, no `optimizer` state, no replay buffer, no `update_step` counter, no RNG state.
- `rl_manager.py` calls `manager.save(model_file)` exactly once at the very end of `train_manager` (line 342). There is no mid-training checkpoint loop and no `resume_from` handling.
- **Verdict:** zero existing resume support. Any longer-horizon work would have to design + implement checkpointing from scratch (model + target + optimizer + replay + step + RNG). That is **out of scope for #103** -- routed to a future issue.
