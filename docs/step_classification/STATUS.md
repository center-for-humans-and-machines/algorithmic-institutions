# Step classification: status and how to resume

Last updated: 2026-11-12, 18:40. This file is the handover for picking the work up on another machine or in another session. Update it whenever the state changes.

## Where everything is

- **Branch:** `annot/step-classification` (from `yannik/handover`). All code, configs, job scripts and docs are committed and pushed here.
- **Plan and spec:** `docs/plans/step-classification-pipeline.md`.
- **Package:** `src/aimanager/annotation/` (README there has every command). Config: `configs/annotation/default.yml`. Tests: `src/aimanager/tests/test_annot_*.py`.
- **Cluster jobs:** `scripts/annotation/cluster/`, documented in `docs/annotation_cluster.md`.
- **Design and progress page:** https://claude.ai/artifact/XjzM7AfruzpWFaSBprrYaC. Its source is kept in `docs/step_classification/design_page.html`; republish from that file after editing.
- **Run data:** `s3://chm-algorithmic-institutions/policy_finder/step_classification/<run_id>/`, synced with `scripts/annotation/s3_sync.py push|pull|ls <run_id>`. Local copies live in `runs/annotation/<run_id>/` (gitignored).
- **Corpus:** read from the 36 refs `origin/policy-finder/pf-2026110{4,5,6}-*` with `git show`; nothing is merged. Rebuild with `python -m aimanager.annotation corpus --run <id>`.

## What is not in git

- **Credentials:** `.env` in the checkout root, gitignored. It needs `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL` (the MinIO keys used by the sibling projects, endpoint `objectstore.hpccloud.mpcdf.mpg.de`), `MINIO_BUCKET=chm-algorithmic-institutions`, and `HF_TOKEN`.
- **Local Python env:** `uv venv .venv-annot` plus the packages listed in the package README. Run with `PYTHONPATH=src`.
- **Cluster access:** `ssh viper`, `ssh dais`, `ssh raven` each need a second factor from a person and then persist for 24 hours. Check with `ssh -O check <host>` before any automated ssh. Never retry ssh in a loop when the master is down: repeated failed logins make the gateways stop answering this machine for a while. If a gateway times out, switch `ProxyJump` between `gate1` and `gate2` in `~/.ssh/config`.
- **On the clusters:** project dir `/ptmp/levinb/algorithmic-institutions-annot` (viper, files removed after 12 weeks without access) and `/dais/fs/scratch/levinb/algorithmic-institutions-annot` (dais). Production weights: `/dais/fs/scratch/levinb/hf_cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017`, revision `08c8657f02291e11b6f6b63e0e16014f8e748518`. Image: `/dais/fs/scratch/levinb/containers/vllm-openai_v0.32.0.sif`.

## Runs so far

| Run id | Where | What | State |
|---|---|---|---|
| `dev0`, `e2e-rev2` | Mac | Fake backend, end to end | Done, local only |
| `viper0` | viper | Trial on a few versions, Qwen3-30B-A3B | Done |
| `viper1` | viper | First full run, both stages, Qwen3-30B-A3B, pass-through review | Done, on S3 |
| `viper2` | viper | Follow-up run after prompt fixes | In progress |
| `smoke-dais` | dais | Job-script tests with Qwen3-30B-A3B | 1-GPU test passes, 2-GPU test queued |
| `dais0` | dais | First boot of DeepSeek V4.1 Flash plus a trial on 6 versions | Not started |

## Next steps

1. Read the viper run report (`runs/annotation/viper1/REPORT.md`, also on S3) and settle the prompts.
2. First boot of the production model on dais (2 B200, tensor-parallel 2; fall back to 4 if memory is short).
3. Production run on dais: stage A, then literature links and codebook review (by Claude in this first run), then stage B, expert check on the held-out rules, join and analysis.
4. Results into new tabs of the design page.

## Decisions taken

- Production model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Development model on viper: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Free-coding sample: one version per lineage (36). Expert-check sample: 30 of the remaining versions. Both are config settings chosen to get the first run going.
- The first run has no stops: codebook review and expert check are done by Claude in the same file formats a person would use. A second run uses real experts.
