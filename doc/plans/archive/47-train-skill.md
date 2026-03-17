# [DONE] Add /train skill and extend /fetch-cluster (#47)

## Task 1: `scripts/train_cluster.sh`

New script following the `remote_test.sh` pattern:

- **SSH check** — reuse same `ssh -O check raven` pattern
- **Sync** — same rsync block as `remote_test.sh` (`.gitignore` filter,
  exclude `.git/`, `artifacts/`, `plots/`, `notebooks/`)
- **Run training** — SSH into Raven and execute:
  ```bash
  cd ~/algorithmic-institutions && source .venv/bin/activate \
    && uv run python src/aimanager/artificial_humans/run.py <config>
  ```
- **Argument**: single required positional arg — config file path
  (relative to repo root)
- **Flags**: `--sync-only` (sync without running), `--no-sync`
  (run without syncing) — matching the `remote_test.sh` style
- **Output**: stream SSH output so SLURM job IDs are visible

## Task 2: `.claude/skills/train.md`

Skill file following the `test.md` pattern:

- Usage: `/train <config_file>`
- Examples with real config paths
- Wraps: `scripts/train_cluster.sh $ARGUMENTS`
- SSH error hint

## Task 3: Extend `scripts/fetch_cluster.sh` with destination

- Accept optional second positional arg: `[local_destination]`
- When provided, sync into that path instead of mirroring remote path
- When omitted, behavior unchanged
- Update usage string accordingly

## Task 4: Update `.claude/skills/fetch-cluster.md`

- Add destination option to usage and examples
