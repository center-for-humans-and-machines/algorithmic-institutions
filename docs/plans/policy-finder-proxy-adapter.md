# [DRAFT] Policy-finder instances through the Claude proxy

## Why

`scripts/policy_finder/{new_instance,run_sweep,run_final}.sh` reach Raven only over an
SSH ControlMaster: rsync to `~/ai-isolated/<branch>`, `ssh … run.py` → `sbatch`, and
`squeue` polling. The Claude proxy has no Raven keys. It reaches one fixed checkout,
`/u/ykeller/algorithmic-institutions`, through the command server. The numerics already
work on that path: validate-rule, sweep sim, GP fit and validation sim were run end to end in
`algoinst.sif` (`docs/reports/handover/06_raven_integration_check.md` §5). Only the
orchestration is missing. The failure points are in `04_policy_finder_orchestration.md` §6.

## Scope

In: running one instance (or a version, `--iterate-on`) from this container, through
check, sweep, fit, final run and commit, with the agent sandboxed on Linux.
Out: the observatory store on `/u/certuer`, and the streamlit dashboard. The HTML export
already runs from the worktrees.

## Steps

1. **Cluster adapter.** Add `scripts/policy_finder/cluster.py`, selected by
   `PF_CLUSTER=ssh|proxy` (default `ssh`, so the manual path is unchanged). It has four
   functions, all on the endpoints the `run_on_cluster` skill documents:
   - `sync(stage)` copies the rule YAML and the sweep configs from the worktree into the
     main checkout. Paths are stem-unique (`configs/managers/rule_based/<stem>.yml`,
     `configs/simulation/policy_finder/<stem>_sweep*.yml`), and the copies stay untracked.
     It then calls `POST /sync_to_cluster`.
   - `submit(config) -> jobid` calls `POST /submit_slurm_job` with
     `-m aimanager simulate <config>`, 1 GPU, 2 CPU, 16G, 01:00:00. On HTTP 409 (the
     4-job cap) it retries with backoff.
   - `wait(jobids)` polls `GET /jobs` every 60 s, as `run_sweep.sh` does now.
   - `fetch(remote_dir, local_dir)` runs `POST /run_in_container` to `cp -r` into
     `/opt/cluster_transfer/pf/<stem>/`, then `POST /sync_from_cluster`, then copies
     from `/var/cluster_transfer`. It checks the size against the ~1 GB guideline.
2. **Scripts.** `run_sweep.sh` and `run_final.sh` call the adapter in place of
   `simulate_cluster.sh`, `ssh` and `rsync`, when `PF_CLUSTER=proxy`. The staged copies
   are removed from the main checkout and from Raven once the instance has fetched.
3. **Host settings.** `new_instance.sh` and `check_instance.sh` take three overrides:
   - `PF_PYTHON` (default `$MAIN/.venv/bin/python`; here `python3`)
   - `PF_BASE` (default `policy-finder-base`; here the branch must be created from
     `origin/policy-finder-base`)
   - `PF_WORKTREE_ROOT` (default `$(dirname MAIN)/policy-finder-worktrees`; `/opt` is not
     writable here)

   `check_instance.sh:112` must keep `PYTHONPATH` instead of replacing it.
4. **Validate in the image.** `check_instance.sh` runs `validate-rule` with
   `/run_in_container` as well, because torch 1.11 rejects code torch 2.5 accepts
   (`th.where(cond, x, 0.0)`).
5. **Linux sandbox.** `instance_settings.py` gets a Linux branch:
   - Session tmp under `/tmp/claude-<uid>/`.
   - Deny reads of the main checkout, the other worktrees and side files, `/var/cc_output`
     and `/var/cluster_transfer`.
   - Deny network to `cc_command_server:8765`, so an agent can never sync or submit.
   - The image needs `bubblewrap` and `socat` (`container/CC_env/Dockerfile`).

   Check that the sandbox engages inside Docker before any agent runs. Bash is
   pre-allowed, so a silent fallback is unsafe.
6. **Tests.** `scripts/tests/test_cluster_proxy.py` covers the adapter against a stub
   HTTP server: sync staging and cleanup, 409 retry, wait, and fetch. Extend
   `test_instance_settings.py` for the Linux paths and `test_policy_finder_check.py`
   for `PF_PYTHON`.
7. **Smoke.** Run one instance end to end with `--sobol-points 16` and few episodes, then
   run one `--iterate-on` version.

## Open questions

- Rule code runs on Raven as `ykeller`, and `rule.py`'s name check does not stop attribute
  access (`th.os`). Should it reject attribute chains outside `th.<function>`?
- Does a 128-point sweep part fit 1 h and 16 GB in the proxy's slurm template? It did on
  the SSH path.
- Who owns the observatory store from now on? Where do `references.json` and the 13
  existing instance branches come from (Cem)?
