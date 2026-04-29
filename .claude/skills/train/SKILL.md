---
name: train
description: Submit a training run on the Raven HPC cluster.
disable-model-invocation: true
argument-hint: <ah|manager> <config_file>
---

Submit a training run on the Raven HPC cluster.

Usage: /train <ah|manager> <config_file>

Training types:
- ah      -- artificial humans (src/aimanager/artificial_humans/run.py)
- manager -- RL manager (src/aimanager/rl_manager.py)

Examples:
- /train ah configs/training/artificial_humans/combined_old_new.yml
- /train manager configs/training/01_rnn_node.yml
- /train --sync-only
- /train --no-sync ah configs/training/artificial_humans/combined_old_new.yml

$ARGUMENTS

Run the remote training script:

```bash
scripts/train_cluster.sh $ARGUMENTS
```

If the SSH connection check fails, tell the user they need to run `ssh raven` in a separate terminal first.
