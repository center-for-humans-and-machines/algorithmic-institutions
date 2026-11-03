# Algorithmic institutions: data and models

Bucket `s3://chm-algorithmic-institutions`. Organized on 2026-11-03 from the code repo
(`center-for-humans-and-machines/algorithmic-institutions`, every branch and its Git LFS
history), the local data exports of its checkout and the oTree repo (`algo-inst-otree`).
`MANIFEST.tsv` lists every file of this layout with its sha256, its size and where it came
from (`git:<commit>:<path>`, `file:<path>`, or `derived` for files this layout builds).
`tools/` holds the scripts that built it.

The 17 keys uploaded on 2026-01-20 were sorted into this layout on 2026-11-03. Three were
byte-identical to files here and were deleted (`models/07_exp2/model/_manager.pt`,
`models/09_exp2_gamma_9/{model/_manager.pt,metrics/.parquet}`). The others moved:

| was | now |
|---|---|
| `experiment_1_mongodb/algoinst/*.json` | `experiment_db/raw_exports/mongodb_json_until_2025-09-04/` |
| `models/07_exp2/metrics/.parquet` | `experiment_1/models/rl_managers_served_by_backend/07_exp2/metrics/` |
| `models/{08,10,11,12}_exp2_gamma_*` | `experiment_1/models/rl_manager_candidates/` |
| `data.csv`, `simulate_multiple_local.ipynb` | `experiment_1/evaluation_sims/simulate_mixed/05_all/` (the output of `05_all.yml`, and an earlier version of the notebook that reads it) |

These files have no other copy (the `08` model is also in the repo's LFS history);
`MANIFEST.tsv` lists them with source `s3:<old key>`, and `tools/moves.tsv` maps old to new keys
(`build_stage.py` reads them from a download of the old keys).

## Layout

```
pre_experiment_1/                 2021-11 to 2022-03, oTree, 4 players + 1 manager, single group
  human_manager_pilots_2021-2022/ test, prepilot, prepilot2-5, pilot1: a human manager
  rule_manager_pilot_2022/        prepilotRand1 ("pilot 2"): a rule-based manager with random params;
                                  pilot_random1_player_round_slim.csv trained the Exp1 clones
experiment_db/                    the experiment DB (one DB for Exp1 and Exp2)
  sessions.tsv                    every session with its phase, dates, groups, rounds
  raw_exports/                    the CSV exports byte for byte, named by their last record
  surveys_without_round_data/     survey rows of sessions that have no round data
  otree_session_ids.txt           the session list kept in the oTree repo
experiment_1/                     AI governor (RL manager) vs human governor, 4 members, 24 rounds
  data/                           the 13 experiment sessions, 169 groups (rounds.csv, surveys/)
  pilot_sessions/data/            the 4 pilot sessions (2024-06 to 2024-08, 11 groups, 16 rounds)
  test_sessions/data/             69 test runs (2024-04 to 2025-04), not participant data
  models/                         see below
  evaluation_sims/                the simulate_mixed runs and plots (data.csv files were never kept;
                                  the one that was is data.csv at the bucket root)
  docs/
experiment_2/                     two groups of 4, group switching every 4 rounds, 24 rounds
  pilot_ai_governor/data/         the pilot with an AI governor: z1awq43i, 2025-09-12, 4 groups
  internal_pilot_coworkers/data/  internal pilot with coworkers as participants: 4tv5t74g, 2025-09-04
  test_sessions/data/             25 test runs (2025-08 to 2026-03), not participant data
  curated_ai_governor_sessions/   group_switching_ah_data_8_agents.csv: the pilot, the internal
                                  pilot and 20 of the test runs together (filter by session)
  human_human_2026/               two human managers: the data the Exp2 human clones are trained on
    data/db/                      the 6 dataset sessions, 50 groups (rounds.csv, surveys/)
    data/curated/                 2group_8agent_50ep.csv (the training data), 35ep, baseline splits
  autoresearch/                   the human-clone autoresearch pipeline
    final_stacks/{levin,frontier}/   the clone stacks in use (models, sim configs, sims)
    final_stacks/training_configs/
    campaign/<experiment>/        each clone experiment branch's own files (see BRANCH.txt)
    notes/                        protocol, summary, review and the per-experiment logs
    handover_reports/             the 2026-10-23 handover write-up
    atlas/report_bundle.html      the Autoresearch Atlas page
```

## The experiment DB

One DB holds every session from 2024-04 on; its exports are cumulative. Each `rounds.csv`
here is one phase's sessions, taken from the newest export that has them
(`raw_exports/rounds_until_2026-03-18.csv`, and for the 2026 human-human dataset sessions
`rounds_human_human_6_sessions_until_2026-04-26.csv`, which has their final revisions).
Phases (`experiment_db/sessions.tsv`):

| phase (`sessions.tsv`) | sessions | groups | dates |
|---|---|---|---|
| Exp1 test (`experiment_1_test`) | 69 | 73 | 2024-04-16 to 2025-04-03 |
| Exp1 pilot (`experiment_1_pilot`) | 4 | 11 | 2024-06-05 to 2024-08-05 |
| **Exp1** (`experiment_1`) | 13 | 169 | 2024-11-14 to 2025-06-24 |
| Exp2 test (`experiment_2_test`) | 25 | 25 | 2025-08-20 to 2026-03-17 |
| Exp2 internal pilot, coworkers (`experiment_2_internal_pilot`) | 1 | 1 | 2025-09-04 |
| Exp2 pilot, AI governor (`experiment_2_pilot_ai_governor`) | 1 | 4 | 2025-09-12 |
| **Exp2 human-human dataset** (`experiment_2_human_human`) | 6 | 50 | 2026-03-16 to 2026-04-26 |

Only sessions named as experiment or pilot sessions are labelled so: the Exp1 sessions and
pilots of `otree_session_ids.txt`, and the Exp2 pilot and internal pilot as Yannik
identified them. Every other session is a test run (the experimenter alone, or partly
complete). The 13 Exp1 sessions are the 10 listed in `otree_session_ids.txt` plus
`kes3nebx`, `d9wukpqo` and `wqehcexq` (June 2025); `hhl7yk2l` has 2 incomplete groups.
Test session `d8sfdujp` has two records of group 2, round 4. Surveys: `manager_assessment.csv` (free text),
`member_assessment.csv` (Likert, single governor, until 2025-09; from the Mongo JSON) and
`member_assessment_gov_a_b.csv` (Likert per governor, 2026).

## Experiment 1 models

| folder | what |
|---|---|
| `rl_manager_deployed/17_exp2_group_payoff_heavy_optimize` | the AI governor of Exp1 ("pgh, gamma 0.98" in the analysis code) |
| `rl_managers_served_by_backend/{04_model,07_exp2,09_exp2_gamma_9}` | the other RL managers the experiment backend (`origin/experiment`) served |
| `rl_manager_candidates/{08_exp2_gamma_8,10_exp2_gamma_5,11_exp2_gamma_0,12_exp2_gamma_7}` | other RL managers of that generation (a gamma sweep), compared in the `evaluation_sims/simulate_mixed` runs (`05_all`, `09_some`) |
| `humanlike_manager_clones/{18,23}_punishment_autoregressive_*` | the human-like governor models the backend served |
| `contribution_clones/{21,22}_*` | the contribution and validity clones the RL manager was trained against |
| `training_configs/` | their `run/` configs |

The `.pt` files load with the code of their time (torch_geometric version), see `docs/`.

## Experiment 2 human clones

`final_stacks/levin` is the stack used for manager sims, RL and the policy finder;
`final_stacks/frontier` is the evaluation-suite reference. Each holds contribution
(GNN + group virtual node + herding copula), switch (`switch_joint_exodus`), validity
(`raven_script_22`) and punisher (multinomial logistic regression with severity copula,
`.joblib`), at their repo paths, plus their sim configs and runs (seeds 42-46 for levin).
The `.pt` files store whole modules and need the code repo (branch `yannik/handover`).

`campaign/` covers the 58 clone experiment branches `origin/auto/*`; the manager
experiments (RL, LLM, rule-based) are not included, nor are two 1 GB confusion-matrix
parquets of `punisher-ar-gnn`.
