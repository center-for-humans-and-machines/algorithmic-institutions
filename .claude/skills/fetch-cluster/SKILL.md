---
name: fetch-cluster
description: Fetch files or folders from the Raven HPC cluster to local.
disable-model-invocation: true
argument-hint: <remote_path>
---

Fetch files or folders from the Raven HPC cluster to local.

Usage: /fetch-cluster <remote_path> [local_destination]

The remote_path is relative to ~/algorithmic-institutions on Raven.
When local_destination is provided, files sync there instead of mirroring the remote path.

Examples:
- /fetch-cluster artifacts/model.pt
- /fetch-cluster artifacts/
- /fetch-cluster configs/training/
- /fetch-cluster temp/training/run1 ./logs/run1

$ARGUMENTS

Run the fetch script:

```bash
scripts/fetch_cluster.sh $ARGUMENTS
```

If the SSH connection check fails, tell the user they need to run `ssh raven` in a separate terminal first.
