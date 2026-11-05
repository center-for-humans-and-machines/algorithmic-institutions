# Large files (data, models, simulation outputs)

This branch does not use Git LFS. Until 2026-11-03, `*.csv`, `*.parquet` and `*.pt` were LFS
objects. They were removed from the tree, and `.gitignore` now ignores those three
extensions. A clone of this branch has the code and configs but none of these files.
Fetch the ones you need from S3.

`docs/large_files.tsv` lists the 264 removed files (171 MB), plus the three experiment 1 clones
(see below): repo path, sha256, size, and
the key of the copy on S3 (`s3://chm-algorithmic-institutions`, see `docs/s3_layout.md`).
That key is usually not the repo path: for example
`artifacts/artificial_humans/raven_script_22/...` is under
`experiment_2/autoresearch/final_stacks/frontier/artifacts/...`.

| | files | MB |
|---|---|---|
| on S3 | 90 | 36 |
| only in the LFS history | 177 | 135 |

Every file is still in Git LFS. Commit `720a7075` (tag `lfs-archive/yannik-handover`) and the
history before it hold the LFS pointers, and GitHub's LFS store has every object
(checked 2026-11-03 by fetching all 264 into a fresh clone). Other branches still use LFS.

The experiment 1 clones the RL manager was trained and evaluated against
(`artifacts/behavioral_cloning/{21_contribution_model_v4,22_contribution_valid_model_v4,23_punishment_autoregressive_v4}`)
were deleted before the LFS removal (18755cfb), so the table did not list them. They were
restored from the LFS objects of b69c994e and are on S3 under `experiment_1/models/`
(`contribution_clones/` and `humanlike_manager_clones/`), checked by sha256 on 2026-11-05.

## Fetch

`scripts/fetch_large_files.py` downloads files from the table into their repo paths and checks
each sha256. Arguments are globs on the repo path; with no argument it fetches everything.
Files already present with the right hash are skipped.

```bash
python scripts/fetch_large_files.py --list                     # what is there, and where from
python scripts/fetch_large_files.py experiments                 # the human datasets
python scripts/fetch_large_files.py 'artifacts/artificial_humans/*'
python scripts/fetch_large_files.py --lfs                       # everything, LFS-only files included
```

- **Default: `s3cmd`** with an `~/.s3cfg` that can read the bucket.
- **`--via command-server`** from the Claude proxy. The command server writes only under
  `/opt/algorithmic-institutions` or `/var/cluster_transfer`.
- **`--lfs`** restores the files that have no S3 copy from commit `720a7075` with
  `git show <commit>:<path> | git lfs smudge`. It needs git-lfs and read access to the
  GitHub repo. Without `--lfs` those files are listed and skipped.

A single file with plain s3cmd, using the key from the table:

```bash
s3cmd get s3://chm-algorithmic-institutions/experiment_2/human_human_2026/data/curated/2group_8agent_50ep.csv \
  experiments/2group_8agent_50ep.csv
```

## What needs which files

- **Human data** (`experiments/`): all on S3 except `pilot_random1_player_round_slim.csv`.
  That file is the training data of the Exp1 clones 21/22. S3 has an older version of it under
  `pre_experiment_1/rule_manager_pilot_2022/`.
- **The final clone stacks** (`24_LEVIN_*`, `24_FRONTIER_*` configs): their models and sims are
  on S3 under `experiment_2/autoresearch/final_stacks/`.
- **LFS only:** the older clone runs (`group_switching_contribution_50ep`,
  `punishment_rnn_edge_50ep_doubled`, `switch_pred_opt*`, `02_*`/`04_*`), all RL manager runs of
  the Exp2 era (`artifacts/manager/0[1-5]*`, `pr6`, `pr7`), most of `plots/simulation/`
  (including `25_LEVIN_run1_ah_zero_pairings_batched/`, the policy finder's reference sim),
  `artifacts/counterfactual/`, `data/baselines/punishment_cv_multinomial_current_contr.csv`
  and the test fixture `src/aimanager/tests/fixtures/episode_raw.csv`
  (`test_baseline_features.py` fails without it).

## Adding new large files

Don't commit them. Upload results worth keeping to S3 with the layout in `docs/s3_layout.md`.
If a small CSV must be tracked (a test fixture, say), add it with `git add -f`.
