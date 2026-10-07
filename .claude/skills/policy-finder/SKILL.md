---
name: policy-finder
description: Run a policy-finder instance (#236) that designs one rule-based manager config in its own sandboxed worktree, then bring the rule back to this checkout. Use when the user asks to run the policy finder, e.g. "run the policy finder with up to 5 params" or "with exactly 3 params".
argument-hint: "[up to|exactly] <N> params"
---

Run a policy-finder instance and bring its rule back.

$ARGUMENTS

1. **Parameters.** "exactly N" -> `--min-params N --max-params N`; "up to N"
   or a bare N -> `--max-params N`; nothing -> the defaults (1 to 4).
2. **Name.** `pf-<YYYYMMDD>-<n>`, with `<n>` the first number for which
   neither branch `policy-finder/<name>` nor
   `../policy-finder-worktrees/<name>` exists.
3. **Launch** in the background (Bash `run_in_background`), and tell the user
   the name and that a run takes a while (tens of minutes):

   ```bash
   scripts/policy_finder/new_instance.sh <name> <param flags> --headless
   ```

   Do not read the instance's files or its report while it runs, and never
   pass it anything about existing or earlier rules: the agent must find its
   own.
4. **Check and commit** once it finishes:

   ```bash
   scripts/policy_finder/check_instance.sh <name> --commit
   ```

   On FAIL, stop: report the offending paths and copy nothing.
5. **Bring the rule back.** Copy
   `../policy-finder-worktrees/<name>/configs/managers/rule_based/<name>.yml`
   to the same path in this checkout. Leave it uncommitted.
6. **Report** from `../policy-finder-worktrees/<name>.report.md`: the
   hypothesis, each parameter with its meaning and sweep range, and the paths
   of the copied rule and of the notes
   (`../policy-finder-worktrees/<name>/notes/policy_finder/<name>.md`, on
   branch `policy-finder/<name>`).

To try a branch before it lands on `policy-finder-base`, set `PF_BASE=<branch>`
on both scripts.
