---
name: simulate
description: Submit a simulation run on the Raven HPC cluster.
disable-model-invocation: true
argument-hint: <config_file>
---

Submit a simulation run on the Raven HPC cluster.

Usage: /simulate <config_file>

Entry point: src/aimanager/simulation/run.py

Examples:
- /simulate configs/simulation/manager_testing/01_compare.yml
- /simulate configs/simulation/ah_testing/01_compare.yml
- /simulate --sync-only
- /simulate --no-sync configs/simulation/manager_testing/01_compare.yml

$ARGUMENTS

Run the remote simulation script:

```bash
scripts/simulate_cluster.sh $ARGUMENTS
```

If the SSH connection check fails, tell the user they need to run `ssh raven` in a separate terminal first.
