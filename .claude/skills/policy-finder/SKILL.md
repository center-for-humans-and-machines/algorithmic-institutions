---
name: policy-finder
description: Run a policy-finder instance (#236) that designs one rule-based manager config, with the agent in its own container from the container spawner, then sweeps it on Raven through the command server and keeps the result on its own branch (#241). Also runs the next version of an existing instance (--iterate-on, #244). Use when the user asks to run the policy finder, e.g. "run the policy finder with up to 5 params", "with exactly 3 params and 512 sobol points" or "iterate on pf-20261008-1".
argument-hint: "[--iterate-on <name>] [--num-params \"up to N\"|\"exactly N\"] [--sobol-points N]"
---

Run a policy-finder instance, or the next version of one: it designs its
rule, is checked, and sweeps it. This runs in the Claude proxy: the agent in
a container from the container spawner (its only sandbox), the sims on Raven
snapshots through the command server. For a whole campaign use
`scripts/policy_finder/campaign.py` instead.

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
   neither branch `policy-finder/<name>` nor `$PF_RUNS/worktrees/<name>`
   exists. With `--iterate-on`, the given name: the launcher refuses an
   instance with uncommitted changes or whose current version has no final
   run yet (`--after-failure` iterates on a failed one). Its new version `k`
   names its files `<name>-v<k>`, the `<version name>` below; for a new
   instance it is `<name>`.
3. **Launch** in the background (Bash `run_in_background`), with `PF_RUNS`
   the folder whose `exports/` is the spawner's `pf_exports` root. Tell the
   user the name and that a run takes a while (the agent tens of minutes,
   then the sweep's queue and jobs):

   ```bash
   PF_RUNS=<pf_runs> scripts/policy_finder/new_instance.sh <name> <param flags> [--sobol-points N]
   PF_RUNS=<pf_runs> scripts/policy_finder/new_instance.sh --iterate-on <name> [<param flags>] [--sobol-points N]
   ```

   It makes the branch and worktree, then `pipeline.py run <name>`: the
   export of what the agent may see, the agent's run, the copy-back,
   `check_instance.sh --commit`, the sweep on a Raven snapshot of the
   commit, the Gaussian process fit, and the final run of the fit's peak and
   the sweep's 5 best points. Each step is recorded in
   `$PF_RUNS/worktrees/<version name>.state.json`
   (`pipeline.py status <name>`). Do not read the instance's files or its
   report while it runs, and never pass it anything about other instances'
   rules: the agent must find its own.
4. **On failure**, report the state's `status` and `error`. `failed` is a
   verdict (a check FAIL with its offending paths, a rule the Raven image
   rejects, a sweep without results). `error` is the environment (network,
   command server, spawner): `pipeline.py run <name>` resumes where it
   stopped, without resubmitting jobs.
5. **Report** from `$PF_RUNS/worktrees/<version name>.report.md`, the state's
   `data.score`, and the sweep dir
   `$PF_RUNS/worktrees/<name>/plots/simulation/policy_finder/<version name>_sweep/`:
   the hypothesis, each parameter with its meaning and sweep range; from
   `landscapes.json`, how much each param moves the pool (`effect`) and the
   params whose peak sits on the edge of their range (`at_edge`); the rule's
   **final best** from `final/sweep.json`: the GP peak (`best`) with its
   params, pool against `ah` (± `pool_se`) and the fit's prediction there
   (`peak.mean` in `landscapes.json`), and next to it the best of the final
   run's points (`best_of_n`); for a new version, against the lineage's best
   so far, the highest GP-peak final of its earlier versions. The sweep's
   own `best` is the top of noisy estimates and reads high: never report it
   as the result. Then the branch `policy-finder/<name>`.

To try a branch other than `policy-finder-proxy`, set `PF_BASE=<branch>` on
the launch.
