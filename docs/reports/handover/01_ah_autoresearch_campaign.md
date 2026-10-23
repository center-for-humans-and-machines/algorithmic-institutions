# Handover 01: the artificial-human autoresearch campaign, machinery and provenance

Scope: how the campaign behind the "Autoresearch Atlas" was run, how the atlas was
built, and where its frontier code ended up. Model internals and the evaluation-suite
internals are covered in other handover notes. Everything below was read from git refs
offline (no `gh`, no network, no cluster). "Inferred" marks conclusions not stated
verbatim in a file.

## TL;DR

- The campaign is 36 Claude-authored experiment PRs, #146 to #181 (2026-08-11 to
  2026-09-16), run by Cem (`cemrtkn`) under the rules in `notes/autoresearch.md`. One
  experiment is one `auto/<slot>-<slug>` branch, one log file `notes/autoresearch_log/<slug>.md`
  and one PR titled `[SUCCESS]` or `[FAIL]`. The verdict came from a single seed-42 sim
  and one `evaluate` on the 21-row suite, judged by two gates.
- Successes were not merged. Later experiments branched off the open `[SUCCESS]` branches,
  which produced two lineages: GNN `main -> #160 -> #165 -> #171 -> #179` (mean 1.099) and
  Gaussian-MLP `main -> #167 -> #170 -> #172 -> #174 -> #177` (mean 1.125). The pre-campaign root scored 1.760.
- The atlas page is the Claude artifact `7a6ef3c8-...`. `bundle_report.py` on
  `origin/autoresearch` (Cem) builds it. I rebuilt it offline from that branch and it
  matched byte for byte. `origin/reports/autoresearch-atlases` (Levin, PR #185) only holds an
  older copy of it, plus two separate "rebaseline" atlases.
- The GNN tip #179 became the base of `autoresearch-optimized-stack` and is on
  `yannik/handover`. The GMLP tip #177 is not: it survives only as a reference sim with its
  outputs. Its code and artifacts live only on `origin/auto/contribution-inflated-gmlp`.
- `24_FRONTIER` is #179 plus the #224 fixes (the punisher reads round t, timeouts recorded as 0).
  `24_LEVIN` is #179 plus the stimulus skip from #181 (a `[FAIL]` PR) plus the #192/#196 punisher.
  Neither one is the stack the atlas scored, and their numbers are not comparable to the atlas numbers.
- Every cluster step of the campaign used the manual SSH path (`train_cluster.sh`,
  `simulate_cluster.sh`, `fetch_cluster.sh`, `sbatch` of `.slurm` wrappers that hardcode
  `~/algorithmic-institutions/.venv`, and per-experiment `~/autoresearch/<slug>` dirs in Cem's
  Raven account `certuer`). Only train and simulate map one-to-one onto the command server.

## 1. Where things live

| ref | author | what it holds |
|---|---|---|
| `origin/autoresearch` (tip `cd227b01`, 2026-09-23) | cemrtkn | long-lived reporting branch: `notes/autoresearch*.md`, `autoresearch_prompts.md`, atlas pipeline `scripts/data_analysis/autoresearch_summary/`, code review `reports/autoresearch_review_{gnn,gmlp,cross}.md`. Not merged into main by design (`notes/autoresearch_summary.md:166-171`) |
| `origin/auto/*` (74 branches) | Claude | 36 corpus branches (all still exist) plus 38 post-corpus ones (#182+, mostly Levin's Sept work on `dev`) |
| `origin/main` (`69451452`) | | frozen. Holds `notes/autoresearch.md` (same text as on `autoresearch`), the #146 merge and its revert (`ab246f4b`), and `doc/merge-recommendations.md` |
| `origin/dev` (`fddf37ad`, Levin) | | integration of PRs #182-#217. Has a later, noise-aware `notes/autoresearch.md` (`1bf38c25`). A reference, not a base (`doc/merge-recommendations.md:9`) |
| `origin/reports/autoresearch-atlases` (`ac362654`, Levin) | | PR #185, merged into `dev` only: `reports/autoresearch_atlas_2026-09-18.html` plus the two rebaseline atlases and their builders |
| `origin/autoresearch-optimized-stack` (`4f80af1b`) | | #179 head plus main plus #224 fixes plus Levin's stack plus #226/#230/#232. Ancestor of `yannik/handover` |

## 2. The harness: what one iteration was

### 2.1 Rules (`origin/autoresearch:notes/autoresearch.md`, identical on main and handover)

- **Metrics and gates** (`:16-35`). From one `evaluation/scores.csv` with 21 rows.
  Gate 1: a declared target row moves into a better band (bands <=1 / 1-2 / 2-5 / >5).
  Gate 2: the 21-row mean rises at most 10% over the baseline mean. Rows <= 1 are reported but do not gate.
- **Evaluation protocol** (`:37-69`). The candidate is swapped into its slot of the
  highest-ranked stack that holds its base model. Ranking is by rows <= 1, then by mean,
  in `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`. Then one
  sim and one evaluation, with no confirmation sweep. On a parent `[SUCCESS]` PR, the parent's
  scores are the baseline (`:206-219`).
- **Legal / illegal / budget** (`:81-121`). One change per experiment. Seeds and episode
  counts are frozen. "Punishment models condition on round t-1" (`:87-89`): post-corpus
  PR #184 later found this assumption wrong (`doc/merge-recommendations.md:53-61`).
  Methods costing more than 3x the base model's training time are banned (`:113-121`).
- **Frozen surface** (`:173-186`): `src/aimanager/evaluation_suite/`, the metric and scoring notes,
  `experiments/`, the scoring parameters, the sim protocol and the stack ranking.
- **Results log** (`:286-309`) has four sections: Declaration, Plan, Results table
  (`date | change | target scores | rows <= 1 | mean | verdict`) and Notes. Merged logs are never edited.

### 2.2 Roles and how a session started

- Roles are defined only in prose, at `notes/autoresearch.md:188-197`. "Fable" opens an experiment
  (research, declaration and plan, loop steps 1-3). Opus orchestrates: it validates the plan,
  tags each step with an implementer (Opus for risky steps, Sonnet otherwise), dispatches one
  subagent per step and commits per step. The tags are visible in the logs, e.g.
  `notes/autoresearch_log/contribution-group-vnode.md:295-301,342-576`.
- `.claude/agents/{architect,researcher,engineer,reviewer,data-analyst,literature-researcher}.md`
  are generic project agents, not campaign-specific. The skills `/train`, `/simulate`,
  `/fetch-cluster` and `/test` (`.claude/skills/*/SKILL.md`) just wrap the SSH scripts.
- No launcher script exists on any ref I searched (no `claude -p` or headless call). Sessions
  were started by hand with prompts like `autoresearch_prompts.md:9-17` ("let's work on rcd on top of
  this work <PR url> ... use this framework notes/autoresearch.md ..."). The roles name the PR
  to build on and the target row. Inferred: sessions were interactive Claude Code runs on Cem's Mac.
- Commit identity was set to `Claude <noreply@anthropic.com>` per worktree (`:232-244`), and pushes went
  through Cem's GitHub account. Rule changes and maintainer rulings are Cem's commits.

### 2.3 Hypothesis to branch to PR

1. Read the rules, prior logs, `[FAIL]` PRs and the base model's deficit profile (`:246-251`).
2. `git worktree add .claude/worktrees/<slug> -b auto/<slot>-<slug> origin/<parent-branch>`
   (`:199-212`). New configs, artifacts and sim dirs carry the slug.
3. Write the declaration and the tagged plan into `notes/autoresearch_log/<slug>.md` and commit.
4. Implement one step per subagent, one commit per step.
5. Train, then (copula stacks only) calibrate and stamp, then simulate, fetch and evaluate (2.4).
6. Verdict from that one evaluation. Open the PR with `--base <parent-branch>`. Body: Hypothesis / Results / Collateral (`:270-283`).

### 2.4 The concrete cluster recipe (manual path, from the #179 log)

`notes/autoresearch_log/contribution-group-vnode.md:275-293` (isolation rules) and `:442-578` (steps).
The whole experiment took about 35 min of cluster time (`:269-274`).

```bash
export AI_REMOTE_DIR='~/autoresearch/contribution-group-vnode'          # Cem's Raven account
scripts/train_cluster.sh --sync-only ah <train cfg>                     # create remote dir
scripts/train_cluster.sh ah configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode.yml
#   -> ssh raven 'python src/aimanager/artificial_humans/run.py <cfg>' -> sbatch job running
#      exec.command `python -m aimanager train-ah {job_file}`         (~10-13 min GPU)
scripts/fetch_cluster.sh artifacts/...          # fetch before the next sync: rsync --delete hazard
sbatch scripts/artificial_humans/calibrate_copula.slurm [BASE] [PARAMS] # rho/phi fit, ~12 min CPU
sbatch scripts/artificial_humans/stamp_copula.slurm [BASE] [PARAMS] [OUT]  # copula fields -> .pt
scripts/simulate_cluster.sh configs/simulation/manager_testing/23_2g8a_switch_joint_exodus_self_gnncopar1_contr_gnn_switch.yml  # control
scripts/simulate_cluster.sh configs/simulation/manager_testing/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml   # candidate
scripts/fetch_cluster.sh plots/simulation/<both dirs>
python -m aimanager evaluate configs/simulation/manager_testing/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml   # local
```

- The control sim must reproduce the parent's `per_round.parquet` sha256 bit for bit before any
  comparison is made (`:555-563`).
- `train_cluster.sh` on the branch (`origin/auto/contribution-group-vnode:scripts/train_cluster.sh:23-25,60-69,84-128`)
  checks `ssh -O check raven`, rsyncs with `--delete`, and in isolated mode borrows
  `~/algorithmic-institutions/.venv` with `PYTHONPATH=<dir>/src SBATCH_EXPORT=ALL`.
- Linear models (the multinomial punisher and the Gaussian-MLP contributor) were trained locally
  with `scripts/baselines/` runners (`notes/autoresearch.md:163-171`).
- Results come back to git: artifacts, the sim's parquet and the scores are committed (LFS) on the
  experiment branch. That is why branches can be scored after the fact.

### 2.5 The rules changed during the campaign (Cem's commits, all on main)

`f81d9b75` (08-12) made a band upgrade mandatory. `ab246f4b` (08-18) reverted the #146 merge on
main, and #160 redid it. `a7a036cb` (08-25) dropped the earlier two-stage protocol, in which Stage 2
was an 8-context sweep. `39896b71` (08-25) set the stack ranking. `c1099d23` (08-25) allowed stacking on
unmerged `[SUCCESS]` branches. `aea7b8f7` (08-27) added the 3x budget. `ed2e38e9` (09-01) handed
orchestration to Opus. `8680db92`/`849e3ca2` (09-01) added `AI_REMOTE_DIR`. `b174f90b` (09-03) added the
10% mean margin, the same day #171 was ruled a SUCCESS after first being logged as a FAIL on gate 2
(`notes/autoresearch_log/switch-joint-exodus.md:222,382-418`). #164 carries a maintainer override
in its title. The atlas keeps each PR's verdict as it was recorded and does not re-judge it
(`notes/autoresearch_summary.md:176-178`).

### 2.6 "Frozen corpus"

`collect.py` and `classify.py` (pipeline steps 1-2) ran once over PRs #146-#181 and must not
be re-run. That was a maintainer ruling (`origin/autoresearch:notes/autoresearch_summary.md:75-82`).
`collect.py:19,34-50` takes every PR numbered 146 or higher whose title starts with `[SUCCESS]`/`[FAIL]`.
For each, `:53-70` fetches `notes/autoresearch_log/<slug>.md` at the PR's head SHA via `gh api`.
The results are frozen as `data/experiments.json` and `data/categories.json`. PRs from #182 on
(Levin's punisher rebaseline, noise floor, RL/rule/LLM-manager arms on `dev`) are outside the corpus,
even where they reuse the `auto/` prefix and the same rules.

## 3. PR to branch map (from `data/experiments.json`)

Spine PRs in bold. A parent of `-` means the PR is based on `main`. Means are the confirmed 21-row means as logged.

| PR | verdict | slot | head branch (`origin/auto/...`) | parent | mean |
|---|---|---|---|---|---|
| 146 | SUCCESS (merged, reverted) | punisher | punisher-severity-copula | - | 1.688 |
| 147 / 149 | FAIL | contribution | contribution-cg-selfdrop / -cg-copula | - | 1.700 / 1.688 |
| 148 | SUCCESS | contribution (lin) | contribution-prev-onehot | - | 1.424 |
| 150 | SUCCESS | switch | switch-herding-copula | - | 1.752 |
| 151-159 | FAIL x9 | contrib / punisher | contribution-gaussian-mlp, punisher-ar-gnn, contribution-{peer-attention, xgboost, cont-target, mlp-regressor, conformity-mixture, type-latent, group-latent} | - | 1.45-1.77 |
| **160** | SUCCESS | punisher | **punisher-severity-copula-v2** | - | **1.688** |
| 161 | FAIL | punisher | punisher-ar-gnn-v2 | - | 1.631 |
| 162 / 163 | FAIL | switch / contrib | switch-herding-copula-v2 / contribution-cg-schedsamp-v2 | 160 | 1.586 / 1.811 |
| 164 | SUCCESS (override) | punisher | punisher-ar-copula | 161 | 1.634 |
| **165** | SUCCESS | contribution | **contribution-herding-copula-v2** | 160 | **1.289** |
| 166 / 168 / 169 | FAIL | switch | switch-herding-copula-v3 / -recal / switch-exodus-count | 165 | 1.723 / 1.306 / - |
| **167** | SUCCESS | contribution | **contribution-gaussian-mlp-v2** | - | **1.615** |
| **170** | SUCCESS | contribution | **contribution-gmlp-group-copula** | 167 | **1.314** |
| **171** | SUCCESS (rule amended) | switch | **switch-joint-exodus** | 165 | **1.304** |
| **172** | SUCCESS | switch | **switch-joint-exodus-gmlp** | 170 | **1.262** |
| 173 / 176 / 178 | FAIL | contribution | contribution-group-size / -arrival-tenure / -pna-aggregation | 171 | 1.352 / 1.323 / 1.351 |
| **174** | SUCCESS | switch | **switch-exodus-k-onehot** | 172 | **1.223** |
| 175 | FAIL | contribution | contribution-size-onehot | 174 | 1.286 |
| **177** | SUCCESS | contribution | **contribution-inflated-gmlp** (tip `c91f9bbf`) | 174 | **1.125** |
| **179** | SUCCESS | contribution | **contribution-group-vnode** (tip `11fe223f`) | 171 | **1.099** |
| 180 / 181 | FAIL | switch / contrib | switch-round-onehot / contribution-punishment-response | 179 | 1.120 / 1.105 |

The atlas counts these as 13 successes and 23 fails. The root is
`plots/simulation/23_2g8a_self_gnn_contr_gnn_switch`, run `lin_multinomial_self`, with mean 1.7596 and
11/21 rows <= 1. The local file is identical to the LFS object on `autoresearch`. Its three models are
listed at `notes/autoresearch.md:62-66`. The PR states above are the `collect.py` snapshot: all
open except #146. I could not check their current state.

## 4. How the atlas page was produced

**Entry point** (on `origin/autoresearch`, local, stdlib only):

```bash
scripts/data_analysis/autoresearch_summary/regenerate.sh   # steps 3-8, prints the step-9 publish call
```

`regenerate.sh:22-26` runs these in order with `PYTHONPATH` set to the pipeline dir:

| step | script | reads | writes |
|---|---|---|---|
| 1 (frozen) | `collect.py` | `gh pr list`, logs at head SHAs | `data/experiments.json` |
| 2 (frozen) | `classify.py --prepare/--merge` | blinded `declaration_method` | `data/classification_input.json`, `data/categories.json` (9-class taxonomy, `:23-33`) |
| 3 | `score_progressions.py` | root `scores.csv` (`:27-30`), cache | `data/spine_scores.json`. On a cache miss: `git fetch`, `gh api pulls/<n>/files`, `git lfs smudge`, match by mean (`:88-113`) |
| 4 | `leaderboard.py` | the same, plus `LABELS`/`FOLDED` (`:34-51`; 11 unique successes, #146 folded into #160 and #172 into #171) | `data/leaderboard_scores.json`, `leaderboard.html` |
| 5 | `stack_visuals.py` | root `evaluation/visuals/*.jpg`, tip branches | `data/stack_visuals.json`, `stack_visuals/{main,gnn,gmlp}/*.jpg`. Always calls `git ls-tree`/`git lfs smudge` on the tip branches (`:34-50`) |
| 6 | `machinery.py` | hand-curated `data/stack_parts.json`, `data/machinery_notes.json` | `machinery.html`, `machinery_pages/` |
| 7 | `build_report.py` | the above plus `notes/evaluation_metric_defs.md` (`:30,82`) | `report.html` |
| 8 | `bundle_report.py` | the above plus `data/review_notes.json` (spoon-feeding flags, "dial" note) | `plots/data_analysis/autoresearch_summary/report_bundle.html` (`:23`) |
| 9 | Artifact tool (agent action) | `data/artifact.json` (url `claude.ai/code/artifact/7a6ef3c8-2ee4-...`) | the published page |

- `spines()` (`score_progressions.py:58-77`) finds the two lineages automatically from the data. The
  "y-axis stack mean" is each PR's logged confirmed mean. The before/after figures are the tip
  branches' own `evaluation/visuals`. The tip sim dirs are found by mean matching:
  `23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch` (#179) and
  `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch` (#177).
- **Verified.** I ran `git archive origin/autoresearch` into a scratch dir, swapped the LFS
  pointer for the identical local root `scores.csv`, and ran steps 3, 4, 6, 7 and 8 offline. Step 5
  was skipped because its outputs are committed. The rebuilt `report_bundle.html` equals the saved
  atlas byte for byte, apart from the artifact host's `<!doctype ...><body>` wrapper. The caches were
  complete, so nothing was fetched. Run the pipeline on `origin/autoresearch`: on another branch,
  `evaluation_metric_defs.md` differs and the legend text changes.
- `origin/reports/autoresearch-atlases:reports/autoresearch_atlas_2026-09-18.html` is an older
  bundle, "previously only in Downloads" (commit `222aa14c`). It has no metric legend, no flags and
  no "dial" tab (these came in `231e1ef7`/`69adcefc`). The rebaseline atlas builders on that branch
  hardcode `/Users/brinkmann/...` worktrees and use `~/Downloads/report_bundle.html` as their template
  (`origin/dev:scripts/reports/build_rebaseline_atlas.py:20-24`).
- The prose summary that `notes/autoresearch_summary.md:179-181` refers to
  (`reports/autoresearch_summary.md`) is not on any ref I checked.

## 5. Lineage of the frontier

```
main ─#160─#165─#171─#179 (11fe223f) ──5c39fcf2 "Merge main into the #179 frontier stack"──┐
                     │                                                                       │
                     └─#181 [FAIL] stimulus skip (dev) ─► cherry-picked as an option cb272375 │
main ─#167─#170─#172─#174─#177 (c91f9bbf)  ── not merged anywhere; reference sim only 1fd1892f│
                                                                                             ▼
autoresearch-optimized-stack: #225 (224-extract-dev-fixes: RCE row, punisher reads round t,
  timeouts, 24_FRONTIER, Levin's stack) -> #229 -> #231 -> #233 ──► policy-finder-base ──► yannik/handover
```

- `git merge-base --is-ancestor` shows that `auto/contribution-group-vnode` (#179) and
  `auto/switch-joint-exodus` (#171) are in `autoresearch-optimized-stack`, `policy-finder-base` and
  `yannik/handover`. No GMLP-lineage branch is in any of them.
  `doc/merge-recommendations.md:7-15` records the decision of 2026-09-24: adopt #160 -> #179 plus
  main, leave out #181, and rebuild the fixes as issue #224.
- On handover, #179's mechanism files (`generic/group_vnode.py`, `copula.py`, `joint_exodus.py`,
  `conditional_bernoulli.py`, `scripts/artificial_humans/contribution_copula_rho.py`) are identical
  to the #179 tip. `generic/graph.py` adds the `stimulus_skip` option (+58 lines), and
  `simulation/linear_ah.py` carries the #224 punisher and timeout changes.
- GMLP on handover: only `configs/simulation/manager_testing/23_2g8a_infl_...yml`, marked
  "REFERENCE ONLY" (lines 1-4), and its outputs. The config points at
  `artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib` and
  `artifacts/artificial_humans/switch_exodus_k_onehot/...`, and both are missing on handover. Running it
  means recovering code and artifacts from `origin/auto/contribution-inflated-gmlp`.

## 6. Levin's stack vs `24_FRONTIER`, and why the numbers moved

Both configs keep the #179 setup (joint-exodus switch, `raven_script_22` valid model, seed 42, 100
episodes) and set `timeout_contribution: 0`:

- `24_FRONTIER_vnode_curpun_...yml:1-4,10-17`: contributor `..._group_vnode_herding_copula` (#179),
  punisher `punishment_multinomial_current_contr_severity_copula.joblib` (#224 C7: reads round t and the timeout flag).
  This is the evaluation-suite reference (`CLAUDE.md:163-166`).
- `24_LEVIN_vnode_skip_timeoutpun_...yml:1-4,10-17`: contributor
  `..._vnode_stimulus_skip_herding_copula` (#181's skip, copula rho 0.0395 / phi 1.0; commit
  `cb272375`), punisher `punishment_multinomial_timeout_severity_copula.joblib` (#192 ceiling
  indicator plus #196 timeout flag, rho stamped at 0.4273; commit `d14c2c3b`). This is the stack for
  manager sims and RL (`CLAUDE.md:176-177`). Commit `0e446a14` reports that it reproduces Levin's `dev` runs per seed.

Means from the local `scores.csv` files (the working-tree files match HEAD's LFS objects):

| run | suite | mean | seeds |
|---|---|---|---|
| #179 as logged (atlas) | 21 rows, t-1 punisher | 1.099 | 42 |
| same #179 sim dir, re-evaluated (`194f6ed7`) | 22 rows | 1.137 | 42 |
| #179 stack re-simulated (`ablation_224/..abl0_old_tdefault*`) | 22 rows | 1.258 +- 0.079 | 42-46 |
| `24_FRONTIER` + `ablation_224/seeds/*curpun_t0*` | 22 rows | 1.183 +- 0.079 | 42-46 |
| `24_LEVIN` + `levin_seeds/` | 22 rows | 1.073 +- 0.056 | 42-46 |
| #177 sim dir, re-evaluated (`1fd1892f`) | 22 rows | 1.166 | 42 |

So the atlas means come from single seed-42 runs under the old suite and the t-1 punisher. Post-corpus
PR #195 (on `dev`) measured a seed SD of 0.047 on the mean and 0.30 on CG, and found that the shipped #179
contributor "ranks first of six on every headline aggregate"
(`origin/dev:notes/autoresearch.md:125-150`). The gains on the spines are larger than that noise,
but the size of each step is a single draw.

## 7. Data flow

```
experiments/2group_8agent_50ep.csv
   │ train-ah (Raven GPU)                 │ scripts/baselines (local sklearn)
   ▼                                      ▼
artifacts/artificial_humans/<x>/model/*.pt   artifacts/baselines/*.joblib
   │ calibrate_copula.slurm -> calibration/copula_params.json
   │ stamp_copula.slurm     -> <x>_herding_copula/model/*.pt (+ .copula.json)
   ▼
configs/simulation/manager_testing/23_2g8a_<slug>_self_<contr>_contr_<switch>_switch.yml
   │ simulate (Raven) -> plots/simulation/<name>/per_round.parquet ; fetch_cluster.sh
   ▼
python -m aimanager evaluate <cfg> (local) -> evaluation/{metrics,scores}.csv, visuals/*.jpg
   ▼
notes/autoresearch_log/<slug>.md results row -> verdict -> PR [SUCCESS]/[FAIL] (base = parent)
   ▼  (all committed on origin/auto/<slug>, LFS)
collect.py (gh) -> experiments.json -> classify -> categories.json            [frozen]
score_progressions / leaderboard / stack_visuals (git + lfs from auto/*) -> data/*.json, jpgs
machinery -> build_report -> bundle_report -> report_bundle.html -> Artifact 7a6ef3c8-...
```

## 8. Manual-path dependencies, and what changes under the command server

| piece | manual-path assumption | under the proxy (`python -m aimanager ...` in `algoinst.sif`) |
|---|---|---|
| `/train`, `scripts/train_cluster.sh ah <cfg>` | ControlMaster `raven`, rsync, remote `.venv`, login-node `run.py` then `sbatch` | `POST /submit_slurm_job`, `-m aimanager train-ah <cfg>`. `params`/`exec` configs are unwrapped (`src/aimanager/cli.py:62-86`). Configs with a `grid`/`include` key are rejected and cannot run on this path |
| `/simulate`, `scripts/simulate_cluster.sh` | same | `-m aimanager simulate <cfg>`. Works as is: the 23/24 configs are flat |
| `/fetch-cluster`, `scripts/fetch_cluster.sh` | rsync back | `/run_in_container` copy to `/opt/cluster_transfer`, then `POST /sync_from_cluster` (run_on_cluster `SKILL.md:56`) |
| `AI_REMOTE_DIR='~/autoresearch/<slug>'`, `.raven_remote_dir` -> `~/ai-isolated/{branch}` | one remote dir per experiment, shared venv via `PYTHONPATH` | one checkout, `/u/ykeller/algorithmic-institutions`. Slug naming is the only isolation, so parallel experiments must not share output paths |
| `calibrate_copula.slurm:28`, `stamp_copula.slurm:29` | `sbatch`, hardcoded `$HOME/algorithmic-institutions/.venv/bin/python` | **breaks**: no `sbatch` in the image and no such venv. Needs a `program_call` of `scripts/artificial_humans/contribution_copula_rho.py ...` (allowed only if the server accepts non-`-m aimanager` calls), or `/run_in_container` on the login node (about 12 min CPU) |
| artifacts reaching Raven | isolated-mode rsync shipped `artifacts/` with `--delete` | `/sync_to_cluster` **excludes `artifacts/` and `plots/`** (`SKILL.md:54`). Committed artifacts (FRONTIER, LEVIN) must already be on Raven |
| linear baselines | trained on macOS | train in the proxy container (sklearn 1.6.1, matching the artifacts' pickled `_sklearn_version`). The Raven image pins sklearn 1.5.2 and numpy <1.24 (`requirements.raven.txt`), while `uv.lock` has 1.6.1 / 1.26.4: a downgrade-unpickle risk |
| `/test`, `scripts/remote_test.sh` | ssh | `/run_in_container` with `python -m pytest ...` |
| `evaluate`, atlas pipeline | macOS, `gh`, `git lfs` | `evaluate` has no torch imports and runs locally. Atlas steps 3, 4, 6, 7, 8 run offline. `stack_visuals.py` and `collect.py` need `git lfs`/`gh`, and neither is installed here |
| bit-identical controls | the same `.venv` every run | the image has a different numpy and sklearn, so expect sims to match only statistically. Inferred, not tested |

**Small Raven tests, in order** (all through the proxy, `cluster: raven`):

1. Check the artifacts are there. `POST /run_in_container`:
   `cd /opt/algorithmic-institutions && sha256sum artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt artifacts/artificial_humans/switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib`.
   Expected local prefixes: `ada5d2aa`, `8a4ae4ad`, `f9a19012`, `e7713cb0`. A missing file or a
   ~130-byte LFS pointer means it has to get onto Raven some other way.
2. Load a punisher in the image: `cd /opt/algorithmic-institutions && python -c "import joblib,sklearn; b=joblib.load('artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib'); print(sklearn.__version__, type(b))"`.
3. Run a 100-episode sim, about 2.5 min: `POST /submit_slurm_job` with `program_call`
   `-m aimanager simulate configs/simulation/manager_testing/24_LEVIN_vnode_skip_timeoutpun_self_gnncopar1_contr_gnn_switch.yml`
   (`time 01:00:00`, `n_cpu 2`, `n_gpu 1`). Pull `per_round.parquet` back and compare its sha256 with
   the committed `7db0bc055be580a2...`. If it differs, run `evaluate` locally and compare with the
   seed-42 mean of 1.051 (seed SD about 0.05). The run overwrites Raven's copy of that `plots/` dir.
   A copy of the config with a new `output_dir` avoids that, but making one is a repo edit and your call.
4. Optional, about 10-13 min GPU: `-m aimanager train-ah configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode.yml`.
   It writes `artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/` on Raven.

## 9. Open questions / could not verify

- I could not read the PR bodies, review threads, current PR states or the GitHub-side merge state
  without `gh`. The `collect.py` snapshot is from 2026-09-16.
- How the agent sessions were launched (interactive, model selection, who typed the prompts) is
  inferred from `autoresearch_prompts.md` and the logs. "Fable" appears only as a role name.
- Whether `/u/ykeller/algorithmic-institutions/artifacts/` holds the FRONTIER and LEVIN artifacts.
  The campaign's own remote dirs (`~/autoresearch/<slug>`, `~/algorithmic-institutions/.venv`) are
  in Cem's account `certuer` (`contribution-group-vnode.md:638-651`), not yours.
- Whether the command server accepts a `program_call` that is not `-m aimanager`. That decides
  whether the copula calibration and stamping steps can run as batch jobs.
- Whether the image's sklearn 1.5.2 unpickles the 1.6.1 joblibs correctly, and how far image sims drift from `.venv` sims.
- The stack-selection matrix `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`
  exists locally. I did not recheck it against the ranking rule.
- The code review (`origin/autoresearch:reports/autoresearch_review_cross.md`) rates "persistence set to
  the maximum three times" as critical (C-X-1) and finds the two tips mergeable. No one has acted on the GMLP half.
