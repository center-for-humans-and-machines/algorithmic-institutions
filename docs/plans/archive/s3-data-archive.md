# [DONE] Save the project's data and models to S3

Task: `docs/tasks/save_and_organize_all_data_to_s3.md`. Bucket: `s3://chm-algorithmic-institutions`
(via the command server's `/s3_*` endpoints). Inventory taken 2026-11-03 from this repo
(all refs, full history, the local LFS store), the Raven checkout and the bucket.

## What the bucket holds today (17 objects, 0.15 GB, all 2026-01-20)

| key | what it is | elsewhere? |
|---|---|---|
| `experiment_1_mongodb/algoinst/Round.json` | DB export of every round, 96 sessions, 2024-04-16 to 2025-09-04 | subset of `data/human_human_rounds_exp2.csv` (same Mongo ids) |
| `experiment_1_mongodb/algoinst/AssessmentManager.json` | manager free-text survey, 111 rows | all in `data/assessment_manager.csv` |
| `experiment_1_mongodb/algoinst/AssessmentMember.json` | member Likert survey, old schema, 968 rows | **only on S3** |
| `models/{07,08,09}_exp2*/model/_manager.pt` | RL managers | in git LFS |
| `models/{10_exp2_gamma_5,11_exp2_gamma_0,12_exp2_gamma_7}/model/_manager.pt` | RL managers | **only on S3** |
| `models/*/metrics/.parquet` | their training metrics | 09 in git LFS, the other five **only on S3** |
| `data.csv` (122 MB) | a simulation output (`ah ... managed by humanlike`, Exp1 evaluation) | not raw data |
| `simulate_multiple_local.ipynb` | the notebook that made it | |

## Inventory against the task's list

1. **Experiment 1 models.** The experiment backend (`origin/experiment`, "Draft sim backend",
   "Remove unneeded models") serves:
   - RL managers `04_model`, `07_exp2`, `09_exp2_gamma_9`, `17_exp2_group_payoff_heavy_optimize`
     ("pgh model", 2024-07-31; the final config in `docs/reports/exp1_manager_training_and_eval.md`)
   - human-like manager clones `18_punishment_autoregressive_v2`, `23_punishment_autoregressive_v4`
   - the contribution clones the RL manager trained against: `21_contribution_model_v4`,
     `22_contribution_valid_model_v4` (both on `experiments/pilot_random1_player_round_slim.csv`)
   - also in history: `08_exp2_gamma_8`, `19_exp2_gamma_7_heavy`, earlier BC/human-like versions.
   Which model ran in which session is set by the experiment app (separate repo), not here.
2. **Pre-Experiment 1 human data** (2021-11 to 2022-03, oTree exports in git LFS history):
   `experiments/{test,prepilot,prepilot2..5,pilot1}` (human manager) and `prepilotRand1`
   (rule-based manager with random params, "pilot 2"), each `main_part_*.csv` + `PageTimes-*.csv`,
   plus the processed `*_player_round*.csv`, `pilot_*pseudo_8_agents.csv`, `combined_pilot_group.csv`.
3. **Experiment 1 data.** The experiment DB exports: S3's Mongo JSON plus local CSV exports
   `data/{round,rounds_exp2,human_human_rounds_exp2}.csv` and `all.csv` (= `rounds_exp2.csv`).
   They are cumulative dumps of one DB that also holds later sessions (see open questions).
   Surveys: manager survey (S3 + `data/assessment_manager.csv`, a superset), member survey (S3 only).
4. **Two-human-manager group switching data:** `experiments/2group_8agent_50ep.csv` (6 sessions,
   50 games; the AH training data), `2group_8agent_35ep.csv`, `baseline/` train/test splits,
   `group_switching/{small,large,final}` raw exports, `group_switching_human_human_group_switching_8_agents.csv`.
   Surveys for these sessions (`data/human_member_assessment.csv`, new schema; the 2026 rows of
   `data/assessment_manager.csv`) are **untracked and gitignored: this checkout is the only copy**.
5. **Autoresearch:** final stacks `24_LEVIN` and `24_FRONTIER` (contribution
   `..._vnode_stimulus_skip_herding_copula` / `..._group_vnode_herding_copula`, switch
   `switch_joint_exodus`, validity `raven_script_22`, punishers
   `punishment_multinomial_{timeout,current_contr}_severity_copula.joblib`), the atlas pipeline
   and bundle, the 75 experiment logs and per-branch artifacts on `origin/auto/*`
   (~2.25 GB, of which two 1 GB confusion-matrix parquets on `auto/punisher-ar-gnn`).

## Proposed layout (alongside the existing keys, nothing moved or deleted)

```
README.md                          what is where, provenance, how to load
MANIFEST.tsv                       key, sha256, bytes, source (git path@commit or local path)
raw/
  pilots_2021-2022/<pilot>/        oTree exports as committed
  experiment_db/<export-date>/     the DB exports, byte for byte, named by their last datetime
pre_experiment_1/                  processed pilot datasets
experiment_1/
  models/behavioral_clones/<run>/  .pt + training config
  models/rl_managers/<run>/        model/ + metrics/ + training config
  models/humanlike_manager/<run>/
  evaluation/                      the Exp1 simulate_mixed outputs
experiment_2/
  human_human/data/                curated datasets (50ep, 35ep, baseline splits)
  human_human/surveys/
  autoresearch/final_stacks/{levin,frontier}/   models + sim config + scores
  autoresearch/campaign/<experiment>/           per-branch artifacts and sim outputs
  autoresearch/atlas/
```

## Steps (after the open questions are answered)

1. Fetch missing LFS objects from GitHub (`git lfs fetch origin <ref> --include=...`).
2. Stage the files under `/var/cluster_transfer/s3_stage/` in the layout above, from git
   objects (`git show`/`git lfs smudge`) and local files, and write the manifest.
3. `/s3_sync_to` with `dry_run` first, then for real; re-list and verify sizes against the manifest.
4. Report which staged files were already on the bucket (sha256 match) and which are new.

## Decisions (2026-11-03)

- 2024-04 to 2025-04 single-group sessions: Exp1 pilots and tests, saved and named as such.
- The 13 multi-group sessions of 2024-11 to 2025-06 (169 groups) are Exp1.
- 2025-08 to 2025-09: Exp2 pilot with an AI governor, saved and named as such.
- 2026: the Exp2 human-human training data (6 sessions, 50 groups).
- Both pre-Exp1 pilot kinds (human manager, rule-based manager) are saved.
- The deployed Exp1 AI governor is 17_exp2_group_payoff_heavy_optimize (the analysis code's
  "pgh, gamma 0.98"); the other models the backend served are saved beside it.
- Autoresearch scope (b): final stacks, atlas, notes and the 58 clone experiment branches,
  without the two 1 GB confusion matrices; no manager (RL/LLM/rule) branches.
- Extras: only the Exp1 evaluation sims. Not saved: Levin's new RL managers, Exp2-era RL
  managers, the Apptainer image, the oTree repo's bonus files and dump.sql.

## Outcome

4107 objects (353 MB) uploaded next to the 17 existing keys, nothing moved or deleted.
A full download matched every sha256 in `MANIFEST.tsv`. The bucket's `README.md` documents the
layout; `tools/build_stage.py` and `tools/campaign.py` rebuild the stage (`temp/s3_stage/`).

Cleanup (same day): the 17 keys of 2026-01-20 were sorted in. Three byte-identical
duplicates were deleted (`models/07_exp2/model/_manager.pt`, `models/09_exp2_gamma_9/*`);
the other 14 were copied to their place in the layout (Mongo JSON to
`experiment_db/raw_exports/mongodb_json_until_2025-09-04/`, `07` metrics beside its model,
`08`/`10`/`11`/`12` to `experiment_1/models/rl_manager_candidates/`, `data.csv` and its
notebook to `experiment_1/evaluation_sims/simulate_mixed/05_all/`), verified by sha256 and
then removed from their old keys. `/s3_rename` rejects keys with a component starting with
`.` (`metrics/.parquet`), so the moves went through a re-upload. Final bucket: 4121 keys,
523 MB, exactly `MANIFEST.tsv`.

Relabeling (same day): only sessions named as pilots or experiments keep those labels;
every other session is a test run. Exp1: 13 experiment sessions, 4 pilots (oTree
`session_ids.txt`), 69 tests. Exp2: pilot with an AI governor `z1awq43i` (4 groups),
internal pilot with coworkers `4tv5t74g`, the 6 human-human dataset sessions, 25 tests
(2025-08 to 2026-03; there was no human-human pilot). The curated
`group_switching_ah_data_8_agents.csv` mixes the pilot, the internal pilot and 20 tests and
moved to `experiment_2/curated_ai_governor_sessions/`. Bucket: 4128 keys, matching `MANIFEST.tsv`.
