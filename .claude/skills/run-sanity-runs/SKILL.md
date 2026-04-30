---
name: run-sanity-runs
description: Run the sanity-snapshot fixtures (3 AH train + 1 sim) on Raven.
disable-model-invocation: true
argument-hint: 
---

Run the sanity-snapshot fixtures on Raven with sanity instrumentation enabled.

Each fixture writes a JSONL log to `src/aimanager/tests/fixtures/_logs/<name>.actual.jsonl` on the cluster. After this completes, pull the logs back with `/fetch-cluster src/aimanager/tests/fixtures/_logs`.

Usage: /run-sanity-runs

Examples:
- /run-sanity-runs
- /run-sanity-runs --no-sync

$ARGUMENTS

Run the script:

```bash
scripts/run_sanity_runs.sh $ARGUMENTS
```

If the SSH connection check fails, tell the user they need to run `ssh raven` in a separate terminal first.
