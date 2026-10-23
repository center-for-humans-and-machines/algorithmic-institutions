# Handover 06: does the new setup run on Raven?

Handover sub-report 6. Scope: what happens when `yannik/handover` (`bb414f5a`)
runs on Raven through the Claude proxy: the command server, `algoinst.sif`,
and the fixed checkout `/u/ykeller/algorithmic-institutions`. Every result
below was measured on 2026-10-23 through the `run_on_cluster` endpoints. The
static analysis of the parts not run is in sub-reports 02, 04 and 05.

## TL;DR

- **The image is fine.** Python 3.9.5, torch 1.11.0+cu113, PyG 2.5.0, sklearn 1.5.2,
  numpy 1.23.5, pandas 2.2.3, pandera 0.22.1. On the login node, 628 of 650 tests
  pass. All 11 failures and errors are files missing from `artifacts/` and `plots/`.
- **Simulation works and reproduces bit for bit.** After two fixes on Raven (below),
  `25_LEVIN_run1_ah_zero_pairings_batched` gives an `aggregates.csv` byte-identical
  to the committed 2026-10-07 reference, and an equal per-round frame (768,000 rows).
  The sklearn 1.6.1→1.5.2 unpickle warning does not change the result.
- **The policy-finder numerics work on the proxy path.** I ran this chain inside
  the image: `validate-rule`, sweep config, an 8-point sweep sim on an A100,
  `fit_surrogate.py`, validation config, and the validation sim (see 5).
- **The integration fails in this order:**
  1. The stale legacy package on Raven shadows `src/aimanager`. Every job fails.
  2. `artifacts/` and `plots/` never reach Raven, so no stack can load.
  3. `train-ah` on any repo AH config fails with `KeyError: 'job_id'`. This bug is in the handover's own `cli.py` change.
  4. The SSH-only orchestration (`new_instance.sh`, `run_sweep.sh`, `run_validate.sh`, copula `.slurm`) has no proxy path yet.
- Fixes 1 and 2 are in place on Raven now, applied by hand. Fix 3 is a code change I have
  not made. With `job_id` set by hand, `train-ah` trains and saves in the image (§6).
- The server also accepts a script path as `program_call`, so the copula calibration and
  stamping steps can run as batch jobs (§7).

## 1. What the Raven checkout looked like

Before any sync, `/u/ykeller/algorithmic-institutions` was the **pre-`src/` layout**:

- the old code: top-level `aimanager/`, `aimanager.egg-info/`, `setup.py`, `requirements*.txt`
- old outputs: `models/`, `models.zip`, `04_all/`, `05_all/`, and `artifacts/` with six old dirs
- a `.git` at `39af134 "Update evaluation."` on `main`, whose packfile "does not match index"

That checkout predates everything in this handover.

`POST /sync_to_cluster` rsyncs without `--delete`, so the new tree is laid *over*
the old one. Old files that the new tree does not have stay in place.

Before the first sync I backed up every old path the sync would overwrite. The backup is
`temp/pre_sync_backup_20261023.tgz` in the Raven checkout: 49 MB, 878 entries, holding
`README.md data djx experiment_analysis experiments notebooks notes run`. `temp/` is
excluded from the sync, so the backup stays put.

## 2. Failure 1: legacy `aimanager/` shadows `src/aimanager`

The slurm template sets `PYTHONPATH=/opt/algorithmic-institutions/src`. But
`python -m` puts the working directory (`/opt/algorithmic-institutions`) **first**
on `sys.path`, so `import aimanager` found the old top-level package:

```
/opt/algorithmic-institutions/aimanager/__init__.py
/opt/venv/bin/python: No module named aimanager.__main__; 'aimanager' is a package and cannot be directly executed
```

This would have failed every `train-ah`, `train-manager` and `simulate` job.

**Applied on Raven.** I moved the two legacy dirs into `temp/legacy_layout/`:
`mv aimanager aimanager.egg-info temp/legacy_layout/`. This is reversible, and afterwards
`aimanager.__file__` is `src/aimanager/__init__.py`.

The other legacy files are still there: `setup.py`, `models*`, `04_all`, `05_all`, and the
corrupt `.git`. They are harmless to imports. `git` on Raven describes the old repo, not
the synced tree.

## 3. Failure 2: artifacts and plots never reach Raven

`/sync_to_cluster` excludes `artifacts` and `plots`, and the exclude matches **at any
depth**. A staging copy under `cc_stage/artifacts/` was dropped as well.

Raven had none of the Levin, FRONTIER or baseline files:
`artifacts/baselines` was absent, and so were all three Levin AH dirs.

That breaks these:
- every simulation config of the new stack
- the RL configs on Levin's stack
- every policy-finder sweep
- 11 tests (`test_eval_*` read `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet`; `test_linear_manager` reads `artifacts/baselines/punishment_multinomial_best_with_contr.joblib`)

**Applied on Raven (workaround).** I staged a local copy of `artifacts/baselines` and
`artifacts/artificial_humans` (33 MB) under an untracked `cc_stage/a/` and synced it.
On Raven I ran `cp -rn cc_stage/a/. artifacts/`, then deleted `cc_stage` on both sides.
`-n` keeps any old Raven artifact with the same path. `artifacts/manager` (117 MB, RL
checkpoints) was not staged.

**Needs a decision.** The proxy needs a supported way to push artifacts, for example
an `artifacts` include list on `/sync_to_cluster`, or a dedicated `/sync_artifacts`
endpoint. The alternative is to accept that artifacts are produced on Raven and staged
this way.

## 4. Simulation: works, reproduces exactly

| | |
|---|---|
| submit | `-m aimanager simulate configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml`, 1 GPU, 2 CPU, 16G, 01:00:00 |
| slurm | job 31441963, `ravg1176`, elapsed 2:58, MaxRSS 9.6 GB, exit 0 |
| log | `Batched: 4 runs x 1000 episodes in 4 batches of up to 1000`; sklearn `InconsistentVersionWarning` (LogisticRegression, StandardScaler pickled with 1.6.1, loaded with 1.5.2) |
| check | pulled back via `/opt/cluster_transfer` + `/sync_from_cluster`. `aggregates.csv` **byte-identical** to the committed file; `per_round.parquet` differs only in writer version (arrow 16.1 vs 21.0). `pandas.DataFrame.equals` is True on 768,000 × 11 |

So for Levin's stack, the image, the copula samplers and the batched path give the same
numbers as the old remote `.venv`. The sklearn downgrade did not change the `ah`
bundle's output. Raising the image's sklearn pin to 1.6.1 is still the safer state.

The run wrote `plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/` on Raven, which did
not exist there before.

## 5. Policy-finder numerics on the proxy path: works

Everything below ran inside `algoinst.sif`, on the login node via `/run_in_container` or
as batch jobs. The test rule is `temp/handover_smoke/hs_threshold.yml`: 2 params,
`punishment = p * (c < c0)`. All outputs went under `temp/handover_smoke/` on Raven,
so no repo path was touched.

| step | where | result |
|---|---|---|
| `python -m aimanager validate-rule temp/handover_smoke/hs_threshold.yml` | login node | `Valid ... 256 distinct design points` |
| `generate_sim_config.py --config ... --sobol-points 8 --n-episodes 100 --out ...` | login node | 8 points × 100 episodes against `ah`, Levin stack |
| `-m aimanager simulate temp/handover_smoke/hs_threshold_sweep.yml` | job 31442043 | exit 0, about 1 min; `sweep.json` with 8 points, `best` = s007 (pool 59.8 ± 4.2) |
| `fit_surrogate.py temp/handover_smoke/out/hs_threshold_sweep` | login node | `landscapes.json`, peak {c0 16.7, p 0.31}: 58.0 ± 3.5 |
| `generate_sim_config.py --validate .../sweep.json --top 2 --n-episodes 200` | login node | 3 points × 200 episodes, seed 1000 |
| `-m aimanager simulate temp/handover_smoke/hs_threshold_sweep_validate.yml` | job 31442132 | exit 0, 1:31, MaxRSS 8.0 GB. `validate/sweep.json`: `best` = s006 (65.9 ± 3.5), s007 65.1 ± 3.3, GP peak 58.1 ± 2.8. This is a toy rule at 200 episodes, but the fresh-seed rerun already reorders the sweep's top points, which is the #238 effect. |

Notes:

- `generate_sim_config.py` refuses a rule outside the repo
  (`Invalid: ... is outside the repo`). A rule must be synced into the checkout first.
- `--validate` always writes its `output_dir` under `plots/simulation/policy_finder/`,
  whatever directory the sweep's `sweep.json` lives in. I redirected it with `sed` for this
  test.
- **torch 1.11 vs 2.5:** `th.where(c < 10, c, 0.0)` raises `RuntimeError: expected scalar
  type float but found double` in the image. torch 2.5 in the proxy container accepts it.
  A rule that passes `validate-rule` locally can therefore crash a whole sweep part on
  Raven. Run `validate-rule` in the image (`/run_in_container`), because its dry run
  executes the code.
- The agent orchestration (`new_instance.sh`, `run_sweep.sh`, `run_validate.sh`) was not
  run. It is SSH-only; sub-report 04 §6 has the failure points and an adapter sketch.

## 6. Failure 3: `train-ah` and the `job_id` gap

| | |
|---|---|
| submit | `-m aimanager train-ah temp/handover_smoke/joint_exodus_smoke.yml`: `switch_predictor/joint_exodus.yml` with 10 epochs, output under `temp/` |
| slurm | job 31442006, exit 1 after 33 s |
| error | `KeyError: 'job_id'` at `src/aimanager/artificial_humans/train.py:263`, via `cli.py:127 dispatch_train_ah` |

**Cause.** The handover commit `77803906` added `unwrap_orchestrator_config`
(`src/aimanager/cli.py:62-85`). It returns `config["params"]`. But the SSH orchestrator
did more than flatten (`artificial_humans/run.py:73-99`). It also set `job_id` from the
labels (`"__".join(f"{k}_{v}" ...)`, which names the saved `.pt`), plus
`output_path`, `temp_dir` and `data_dir`. No config in `configs/training/artificial_humans/`
sets `job_id`. Two things therefore fail on the proxy path: every AH training, and the example
in the proxy's `run_on_cluster` skill. `test_cli_orchestrator_config.py` tests only the
unwrap.

**Check.** The same config with `job_id: architecture_mlp+rnn+edge__dataset_50ep_doubled`
added (`temp/handover_smoke/joint_exodus_smoke_jobid.yml`, job 31442101):
**exit 0** after 2:05 (wandb offline run, MaxRSS 8.0 GB, which is the full 8 GB that `exec.memory` requests). It wrote `model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`, `metrics/` and `confusion_matrix/` under `temp/handover_smoke/switch_joint_exodus/`. With `job_id` set, the joint-exodus training path runs end to end in the image. Nothing else is missing.

**Fix (not applied).** In `unwrap_orchestrator_config`, when `params` is returned and has
no `job_id`, set it as `run.py:97` does. The test would then assert that a repo config
reaches `train.main` with `job_id`.

## 7. Scripts outside `python -m aimanager`

The copula steps of the GNN contributor (`calibrate_copula.slurm`, `stamp_copula.slurm`)
call `$HOME/algorithmic-institutions/.venv/bin/python`, which does not exist in the image.
They are `sbatch` wrappers. One check:

- `program_call: "scripts/artificial_humans/contribution_copula_rho.py --help"`, `launcher: python`
  (job 31442159): **accepted and ran** (exit 0, printed the script's help). A script path works as a `program_call`. `calibrate_copula.slurm` and `stamp_copula.slurm` can therefore become two jobs, `scripts/artificial_humans/contribution_copula_rho.py --model <base> --roundtrip --preflight --write-params <json>` and `scripts/artificial_humans/make_contribution_copula_artifact.py ...`. They need no `.venv` and no `sbatch`. The same holds for any `scripts/baselines/*` step that needs the Raven stack.

## 8. State left on Raven

| path (in `/u/ykeller/algorithmic-institutions`) | what |
|---|---|
| everything outside `artifacts/ plots/ temp/ .git` | `yannik/handover` working tree as of this session, including real LFS content (this container has no git-lfs; files are materialized) and `docs/reports/handover/` |
| `temp/pre_sync_backup_20261023.tgz` | the old files that the first sync overwrote |
| `temp/legacy_layout/{aimanager,aimanager.egg-info}` | the moved legacy package |
| `artifacts/baselines/`, `artifacts/artificial_humans/<new dirs>` | staged copies of the local artifacts (no old file overwritten) |
| `plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/` | the reproduction run |
| `temp/handover_smoke/` | test rule, configs and outputs |

## 9. Recommended next steps

1. Fix the `job_id` gap in `cli.py` (small change plus a test). Re-run the joint-exodus smoke job.
2. Decide how artifacts reach Raven (see 3). Until then, stage them by hand as above.
3. Raise `scikit-learn` in `container/CC_env/requirements.raven.txt` to 1.6.1 (it supports
   Python 3.9). Rebuild the image when convenient. Today's results show the downgrade is
   harmless for the Levin bundle, but not for every bundle.
4. Run `validate-rule` for agent-written rules in the image, not locally.
5. Before the policy-finder agents can run here, add the proxy adapter for
   `run_sweep.sh` and `run_validate.sh` (sub-report 04 §6), and rewrite the sandbox for
   Linux. `rule.py`'s name check is not a sandbox (`th.os` is reachable; sub-report 05), and
   rule code runs on Raven as `ykeller`.
