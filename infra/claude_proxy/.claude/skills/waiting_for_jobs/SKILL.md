---
name: waiting_for_jobs
description: Block on a Raven slurm job's state via the CC external command server. Use whenever the user asks to wait until a previously submitted slurm job starts running, finishes, is cancelled, or otherwise leaves the queue, instead of busy-polling `/jobs` yourself.
---

# Waiting on a slurm job

`wait_for_job.sh` polls `http://cc_command_server:8765/jobs`. It is on `PATH`. Use it instead of polling `/jobs` from the agent. It sleeps 2 minutes between cluster polls.

For a training run of many hours, do not block until it finishes on the first attempt. Sleep for a few hours and check the log, so a broken job is cancelled early.

## Usage

```bash
wait_for_job.sh http://cc_command_server:8765 raven <JOB_ID> [WAIT_UNTIL_DONE]
```

- `JOB_ID` — slurm job id from the submit log (`12345` or `12345_0`).
- `WAIT_UNTIL_DONE` — optional. Absent: exit once the job is no longer pending. Present (any non-empty value): keep polling while it runs, and exit on stopping or disappeared.

The script prints one of:

- `wait_for_job: job <id> is running`
- `wait_for_job: job <id> is stopping (state=<code>)`
- `wait_for_job: job <id> has disappeared from the queue`

Exit 0 on those, 2 on bad arguments, 3 if `curl`, `jq`, or `awk` is missing.

Default mode is for an interactive allocation (`/request_single_gpu`) before `/run_in_job`. Wait-until-done is for a batch job whose outputs you need next.
