Fetch files or folders from the Raven HPC cluster to local.

Usage: /fetch-cluster <remote_path>

The remote_path is relative to ~/algorithmic-institutions on Raven.

Examples:
- /fetch-cluster artifacts/model.pt
- /fetch-cluster artifacts/
- /fetch-cluster configs/training/

$ARGUMENTS

Run the fetch script:

```bash
scripts/fetch_cluster.sh $ARGUMENTS
```

If the SSH connection check fails, tell the user they need to run `ssh raven` in a separate terminal first.
