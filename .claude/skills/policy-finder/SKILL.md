---
name: policy-finder
description: Run a policy-finder instance (#236) that designs one rule-based manager config in its own sandboxed worktree, then sweeps it on Raven and keeps the result on its own branch (#241). Use when the user asks to run the policy finder, e.g. "run the policy finder with up to 5 params" or "with exactly 3 params".
argument-hint: "[up to|exactly] <N> params"
---

Run a policy-finder instance: it designs its rule, is checked, and sweeps it.

$ARGUMENTS

1. **Parameters.** "exactly N" -> `--min-params N --max-params N`; "up to N"
   or a bare N -> `--max-params N`; nothing -> the defaults (1 to 4).
2. **Name.** `pf-<YYYYMMDD>-<n>`, with `<n>` the first number for which
   neither branch `policy-finder/<name>` nor
   `../policy-finder-worktrees/<name>` exists.
3. **Launch** in the background (Bash `run_in_background`). The sweep needs
   the SSH ControlMaster to Raven: check `ssh -O check raven` first, and if
   it fails ask the user to run `ssh raven`. Tell the user the name and that
   a run takes a while (the agent tens of minutes, then the sweep's queue
   and jobs):

   ```bash
   scripts/policy_finder/new_instance.sh <name> <param flags> --headless
   ```

   It runs the agent, then `check_instance.sh <name> --commit`, then
   `run_sweep.sh <name>`, which commits the sweep on `policy-finder/<name>`.
   Do not read the instance's files or its report while it runs, and never
   pass it anything about existing or earlier rules: the agent must find its
   own. Nothing of the instance is copied into this checkout.
4. **On failure**, stop and report: a check FAIL with its offending paths
   (no sweep runs), or the sweep step that failed. A sweep that failed after
   a passing check can be rerun with `scripts/policy_finder/run_sweep.sh <name>`.
5. **Report** from `../policy-finder-worktrees/<name>.report.md` and
   `../policy-finder-worktrees/<name>/plots/simulation/policy_finder/<name>_sweep/sweep.json`:
   the hypothesis, each parameter with its meaning and sweep range, the best
   params and their pool against `ah` (± `pool_se`), and the branch
   `policy-finder/<name>`. The observation app shows the rest.

To try a branch before it lands on `policy-finder-base`, set `PF_BASE=<branch>`
on the launch.
