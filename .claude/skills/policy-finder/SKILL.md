---
name: policy-finder
description: Run a policy-finder instance (#236) that designs one rule-based manager config in its own sandboxed worktree, then sweeps it on Raven and keeps the result on its own branch (#241). Also runs the next version of an existing instance (--iterate-on, #244). Use when the user asks to run the policy finder, e.g. "run the policy finder with up to 5 params", "with exactly 3 params and 512 sobol points" or "iterate on pf-20261008-1".
argument-hint: "[--iterate-on <name>] [--num-params \"up to N\"|\"exactly N\"] [--sobol-points N]"
---

Run a policy-finder instance, or the next version of one: it designs its
rule, is checked, and sweeps it.

$ARGUMENTS

1. **Arguments.**
   - `--num-params "up to N"` -> `--max-params N`; `--num-params "exactly N"`
     -> `--min-params N --max-params N`; left out -> the defaults (1 to 4).
     Free text ("up to 5 params", "exactly 3") reads the same way.
   - `--sobol-points N` -> `--sobol-points N`: the sweep's design size, a
     power of two; left out -> 256. The sweep runs in parts of at most 125k
     episodes (500 per point), so 256 points take 2 jobs, 512 take 3.
   - `--iterate-on <name>` (free text: "iterate on <name>") ->
     `--iterate-on <name>`: the next version of that existing instance, on
     its own branch and worktree. Left out `--num-params` then means any
     number of params from 1, the agent's choice; left out `--sobol-points`
     the previous version's.
   Anything else, or N not a positive integer: stop and ask.
2. **Name.** `pf-<YYYYMMDD>-<n>`, with `<n>` the first number for which
   neither branch `policy-finder/<name>` nor
   `../policy-finder-worktrees/<name>` exists. With `--iterate-on`, the
   given name: the launcher refuses an instance with uncommitted changes or
   whose current version is not validated yet. Its new version `k` names its
   files `<name>-v<k>`, the `<version name>` below; for a new instance it is
   `<name>`.
3. **Launch** in the background (Bash `run_in_background`). The sweep needs
   the SSH ControlMaster to Raven: check `ssh -O check raven` first, and if
   it fails ask the user to run `ssh raven`. Tell the user the name and that
   a run takes a while (the agent tens of minutes, then the sweep's queue
   and jobs):

   ```bash
   scripts/policy_finder/new_instance.sh <name> <param flags> [--sobol-points N] --headless
   scripts/policy_finder/new_instance.sh --iterate-on <name> [<param flags>] [--sobol-points N] --headless
   ```

   It runs the agent, then `check_instance.sh <name> --commit`, then
   `run_sweep.sh <name>`, which commits the sweep on `policy-finder/<name>`
   with its Gaussian process fit (`landscapes.json`), then runs and commits
   the validation sim (`run_validate.sh`), then a headless `/policy-finder-summarizer`, which places the rule in the
   observation app's Summary (`../policy-finder-worktrees/<version name>.summary.md`).
   Do not read the instance's files or its report while it runs, and never
   pass it anything about other instances' rules: the agent must find its
   own. A new version sees only its own instance's earlier versions, in its
   worktree. Nothing of the instance is copied into this checkout.
4. **On failure**, stop and report: a check FAIL with its offending paths
   (no sweep runs), or the sweep step that failed. A sweep that failed after
   a passing check can be rerun with `scripts/policy_finder/run_sweep.sh <name>`,
   a validation that failed after a committed sweep with
   `scripts/policy_finder/run_validate.sh <name>`.
5. **Report** from `../policy-finder-worktrees/<version name>.report.md` and
   the sweep dir
   `../policy-finder-worktrees/<name>/plots/simulation/policy_finder/<version name>_sweep/`:
   the hypothesis, each parameter with its meaning and sweep range; from
   `landscapes.json`, how much each param moves the pool (`effect`) and the
   params whose peak sits on the edge of their range (`at_edge`); the rule's
   **final best** from `validate/sweep.json` only: its params, pool against
   `ah` (± `pool_se`) and which point it was (`<version name>_gp`, the
   Gaussian process's peak, or a measured `<version name>_s<index>`); for a
   new version, against the lineage's best so far, the highest final best of
   its earlier versions (each one's `validate/sweep.json`; higher is an
   improvement, whatever the margin). The sweep's own `best`
   is the top of noisy estimates and reads high: never report it as the
   result. Then the branch
   `policy-finder/<name>`; and from `<version name>.summary.md`, the categories the
   rule was placed in (or that the summary failed: then run
   `/policy-finder-summarizer`). The observation app shows the rest.
6. **Republish the shared page**, which the headless summary cannot: step 4
   of `/policy-finder-summarizer` (export the fragment, publish it to the
   shared artifact). With several instances running, do it once, after the
   last one finishes.

To try a branch before it lands on `policy-finder-base`, set `PF_BASE=<branch>`
on the launch.
