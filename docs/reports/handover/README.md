# Handover: how the autoresearch clones and the policy finder were produced

For: Yannik (taking over from Cem). Written on 2026-10-23 on `yannik/handover`
(`bb414f5a`). The report covers the two result pages handed over:

- the **Autoresearch Atlas**, which holds the new human clones (claude.ai artifact `7a6ef3c8-…`)
- the **Policy Finder observatory**, the rule-based managers, snapshot of 2026-10-15 (artifact `be0d6888-…`)

It also records how this branch runs on Raven through the Claude proxy. This page is
the summary. Details, with `path:line` references, are in the sub-reports:

| # | sub-report | read it for |
|---|---|---|
| 01 | [AH autoresearch campaign](01_ah_autoresearch_campaign.md) | the experiment protocol, the PR→branch map of #146–#181, how the atlas page is built |
| 02 | [AH frontier stacks](02_ah_frontier_stacks.md) | model code, flags, training configs and artifacts per role × stack |
| 03 | [Evaluation suite](03_evaluation_suite.md) | the 21/22 score rows, the noise-ceiling score, seed noise |
| 04 | [Policy finder: orchestration](04_policy_finder_orchestration.md) | instances, the agent and its sandbox, the dashboard, where results live |
| 05 | [Policy finder: rule, sweep, surrogate](05_rule_manager_sweep.md) | rule schema, sweep sim, definition-4 score, GP fit, validation |
| 06 | [Raven integration check](06_raven_integration_check.md) | what ran on Raven today, and where it broke |

## 1. The new human clones (Autoresearch Atlas)

**What produced them.** A campaign of 36 Claude-written experiments, PRs #146–#181,
run 2026-08-11 → 09-16 and supervised by Cem. The rules are in `notes/autoresearch.md`:
gates at `:16-35`, protocol at `:37-69`, roles at `:188-197`. One experiment works like this:

- It is one branch, `origin/auto/<slot>-<slug>`, and makes **one** change to one slot:
  contribution, switch or punisher.
- Its log is `notes/autoresearch_log/<slug>.md`, and it ends in a PR titled `[SUCCESS]`/`[FAIL]`.
- The verdict comes from a single training run, a single seed-42 simulation (100 episodes)
  and one `python -m aimanager evaluate` on the 21-row suite.
- Two gates decide: the target row moves into a better band, and the 21-row mean rises by
  less than 10%.

Successes were not merged. Later experiments branched off them, which produced two lineages:

```
GNN    main → #160 → #165 → #171 → #179      mean 1.099  (root stack 1.760)
GMLP   main → #167 → #170 → #172 → #174 → #177   mean 1.125
```

**The per-experiment pipeline** (manual SSH path, as the campaign ran it; 01 §2.4):

```
experiments/2group_8agent_50ep.csv (50 games, flip-doubled)
 ├─ train-ah  (Raven GPU)  configs/training/artificial_humans/{contribution,switch_predictor}/*.yml
 │     → artifacts/artificial_humans/<x>/model/<job_id>.pt
 │     → calibrate_copula.slurm + stamp_copula.slurm   (contributor copula: rho, phi stamped onto the .pt)
 ├─ scripts/baselines/run_baseline_cv.py → inspect_best_model.py --save-best → punishment_copula_rho.py
 │     (local sklearn) → artifacts/baselines/punishment_*_severity_copula.joblib
 ├─ simulate (Raven)  configs/simulation/manager_testing/23_2g8a_<...>.yml → plots/simulation/<name>/per_round.parquet
 ├─ evaluate (local)  → evaluation/{metrics,scores}.csv   → results row in the log → PR verdict
 └─ atlas: origin/autoresearch:scripts/data_analysis/autoresearch_summary/regenerate.sh → report_bundle.html
```

**What the models are, and where the code lives on this branch** (02 §1):

| role | GNN stack (#179) | switched on by |
|---|---|---|
| contribution | `GraphNetwork` (`generic/graph.py:130`), your node+edge+rnn model, plus the group virtual node (`generic/group_vnode.py:68`), plus the herding copula at sampling time (`generic/copula.py:48`) | `model_args.group_vnode: True`; copula fields stamped after training, not learned |
| switch | your mlp+rnn+edge switch model plus `JointExodusHead` (`generic/joint_exodus.py:168`), then conditional Bernoulli (`generic/conditional_bernoulli.py:158`) | `model_args.joint_exodus: True`, `joint_exodus_switch_every: 4` |
| punisher (`ah`) | **not a GNN**: a 31-level multinomial logistic regression with a severity copula (`simulation/linear_ah.py:286,343`), loaded as manager `type: linear` | bundle key `copula_rho` |
| validity (timeouts) | the legacy `raven_script_22` model, unchanged since February | — |

The frontier GNNs are your original configs with these flags added, and the hyperparameters
are unchanged. The `.pt` files store whole modules by import path, so an edit to those
classes changes what the shipped models do (02 §6.4).

**What ended up on your branch** (01 §5):

- The **#179 lineage is here.** It became the base of `autoresearch-optimized-stack`, and from there of `policy-finder-base` and `yannik/handover`.
- The **GMLP lineage (#177) is not.** Its code, configs and artifacts exist only on `origin/auto/contribution-inflated-gmlp`, which is divergent code: it has no vnode, no stimulus skip and no timeout handling. Only a "REFERENCE ONLY" sim config remains here.
- The two stacks in use today are both **#179 variants with post-campaign fixes**:
  - `24_FRONTIER_*` is the evaluation-suite reference. It adds the #224 punisher, which reads round *t* and the timeout flag.
  - `24_LEVIN_*` is used for manager sims, RL and the policy finder. It adds #181's stimulus skip (a `[FAIL]` PR in the atlas) and the #192/#196 punisher.

**How to read the atlas numbers** (01 §6, 03):

- **They come from single runs.** Each is one seed-42 run on the old 21-row suite, with a
  punisher that read round *t−1*, an assumption later found wrong.
- **Noise is large.** Across sim seeds the mean has SD ≈ 0.08 and single contribution rows
  ≈ 0.3. Across training seeds the mean has SD ≈ 0.05.
- **What survives the noise.** The big band jumps (CG 9.8→0.9, PD 2.9→1.5, SC 2.7→1.3) are
  real. The gap between the two tips is not.
- **Today's stacks, re-scored** (22 rows, seeds 42–46): LEVIN 1.073 ± 0.056,
  FRONTIER 1.183 ± 0.079, #179 1.258 ± 0.079.
- **The stack mean is not computed by suite code.** It is the unweighted mean of the
  `score` column. `evaluation_sweep.py` still hard-codes 21 rows and drops RCE (03).

## 2. The rule-based manager search (Policy Finder)

**What produced it.** One *instance* is one branch, `policy-finder/<name>` (the `pf-YYYYMMDD-n`
names), checked out as a worktree. A version (`--iterate-on`) is the next attempt on the same
branch, and a lineage is an instance plus its versions. 04 has the full lifecycle.

```
new_instance.sh <name> [--iterate-on] --headless            (host machine, outside the sandbox)
 ├─ claude -p --agent policy-finder-host  (Sonnet relay) → policy-finder subagent (Opus, sandboxed:
 │    may write only configs/managers/rule_based/<stem>.yml, notes/policy_finder/<stem>.md, scripts/policy_finder/<name>/)
 ├─ check_instance.sh --commit        validate-rule + diff/notes checks, then commits for the agent
 ├─ run_sweep.sh                      generate_sim_config.py → 256 Sobol points × 500 episodes vs `ah`,
 │                                    2 sim parts on Raven → merge_sweeps.py → fit_surrogate.py (GP) → commit
 ├─ run_validate.sh                   GP peak + top-5 measured points, 2000 episodes, seed 1000 → final `best`
 └─ claude -p /policy-finder-summarizer → summary.json; export_html.py → the observatory page
```

- **The rule** is a YAML file: `params`, `sweep_config` ranges, optional `constraints`, and `code`,
  a torch snippet `exec`'d per round over the batch. The schema and the checks are in
  `src/aimanager/manager/rule.py:6-29,423`.
- **The sim** is `python -m aimanager simulate <sweep.yml>` with `sweep: true` and
  `episode_batch_size: 1000`, on Levin's stack (`simulation/simulate.py`, `sweep.py`).
- **The score** is definition 4 (`simulation/pool_scores.py`): the rule's group common pool
  1.6·Σc − Σp per round against `ah` in the other group.
- **The quoted "± 1.1"** is the plain standard error of the 2000-episode validation. It does
  not correct for picking the best of about 6 rerun points (05).
- **The reference numbers** 56.9 / 69.8 / 59.5 come from `references.json` on Cem's side and
  are not in the repo. As a check, the #226 run-1 data here gives `ah` 56.0 and `zero` 60.5 (05).

**Where the results live.** None of the 13 instance branches is in this clone or in the last
fetch (2026-10-23, 117 remote branches). Paths in the snapshot place them in Cem's local clone
(`/Users/ertuerkan/Desktop/`), with the sim outputs under `/u/certuer/ai-isolated/` on Raven. The snapshot HTML is
a partial backup: it holds all 32 rule files, the notes and the plots, but no per-round data.
To reproduce or continue it you need Cem to push the `policy-finder/*` branches (with LFS)
and the side files (04 §4).

**No rule configs on this branch.** `configs/managers/rule_based/` is empty on
`policy-finder-base` by design (#235). The old `sigmoid`/`decay` rules on
`autoresearch-optimized-stack` use a schema that today's `validate-rule` rejects (05).

## 3. Does the new setup run? (Raven via the proxy)

Measured today through `run_on_cluster`; full log in 06.

| step | status |
|---|---|
| command server, SSH, `algoinst.sif` (py 3.9.5, torch 1.11.0+cu113, PyG 2.5.0) | ok |
| **① `python -m aimanager …`**: stale legacy `aimanager/` in `/u/ykeller/algorithmic-institutions` shadows `src/` (cwd is first on `sys.path`) | **broke every job**. Fixed on Raven by moving it to `temp/legacy_layout/`. The old files the sync overwrote are backed up in `temp/pre_sync_backup_20261023.tgz` |
| test suite on the login node | 628 passed, 15 skipped, 11 failed/errored, all on missing `artifacts/`/`plots/` files |
| **② artifacts**: `/sync_to_cluster` excludes `artifacts` and `plots` at any depth, and Raven had no stack artifacts | **broke every new-stack sim**. Worked around by staging 33 MB under another dir name. A supported path is still needed |
| `simulate` on Levin's stack (`25_LEVIN_run1_ah_zero_pairings_batched`) | ok. **Bit-identical** to the committed reference despite sklearn 1.6.1→1.5.2 |
| policy-finder numerics: validate-rule → sweep sim → GP fit → validation sim | ok, end to end in the image |
| **③ `train-ah` on any repo AH config** | **broke**: `KeyError: 'job_id'` (`train.py:263`). The handover's `cli.py` unwrap drops the `job_id` that `run.py:97` used to add. With `job_id` set it trains and saves |
| script path as `program_call` (copula calibrate/stamp) | accepted. The `.slurm` wrappers are not needed |
| **④ `new_instance.sh` / `run_sweep.sh` / `run_validate.sh`** | not run. They are SSH/rsync/sbatch only, and the first failures are at `new_instance.sh:112` and `run_sweep.sh:55`. They need a proxy adapter and a Linux sandbox rewrite first (04 §6) |

**First failing step today.** On Raven, the code and the image work once ① and ② are dealt with.
What fails next is ③, in the branch's own code, then ④, the orchestration that was never
ported to the proxy.

## 4. Things worth knowing before you change anything

- **The modified files are not edits.** The ~260 files `git status` shows as modified are LFS
  content: this container has no git-lfs, and every one hashes to its committed pointer (02, 03).
  Install git-lfs before committing data or artifacts from here.
- **Rule code is not sandboxed.** `rule.py`'s load-time check looks only at names. `th.os` is
  reachable, and agent-written rule code runs on Raven as `ykeller` (05). The agent-side sandbox
  (`instance_settings.py`) is written for macOS paths (04).
- **torch 1.11 is stricter than torch 2.5.** `th.where(cond, x, 0.0)` fails in the image and
  passes in this container. Run `validate-rule` in the image (05, 06).
- **The sklearn pin is behind.** `requirements.raven.txt` pins sklearn 1.5.2, while the lock and
  the bundles use 1.6.1. That was harmless for the Levin bundle; raise the pin when the image is rebuilt.
- **Two sim seeds count as two separate measurements.** Compare stacks over seeds 42–46, and
  compare managers on the 1000-episode batched runs.

## 5. Suggested next steps

1. Fix the `job_id` gap in `cli.py` and add a test (06 §6).
2. Agree how artifacts reach Raven: change the sync, or stage them by hand.
3. Ask Cem for the `policy-finder/*` branches, `references.json`, and read access to (or a copy of)
   `/u/certuer/ai-isolated/` and the observatory store.
4. Decide whether the GMLP lineage matters. If it does, it has to be ported from
   `origin/auto/contribution-inflated-gmlp`.
5. Write a plan in `docs/plans/` for the proxy adapter of the policy-finder scripts (04 §6) before
   running any agent from this container.
