# Step classification: status and how to resume

Last updated: 2026-11-13, 01:15. This file is the handover for picking the work up on another machine or in another session. Update it whenever the state changes.

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
| `viper0` | viper | Trial on a few versions, Qwen3-30B-A3B | Done, on S3 |
| `viper1` | viper | First full run, both stages, Qwen3-30B-A3B, pass-through review, simulated expert check | Done, on S3, with `REPORT.md` |
| `viper2` | viper | Same corpus after the prompt fixes of commit 732e137d; stage A, pass-through review and stage B in one job | Done, on S3; compared in viper1's `REPORT.md` |
| `viper3` | viper | Design decisions of commit 0e990dd9: positive ideas, consolidated two-level codebooks, coverage judged with the draft and a prevalence cap, repeated judge with test-retest reliability, basis codebook | Done (jobs 13081258, 13082056), on S3 with `REPORT.md`; judge test-retest 0.91 (kappa 0.79); fixes f9827c51 and 68047187 not yet run |
| `viper4` | viper | Commit 47631873: basis consolidation, every code placed in a family, merge reasons, unreliable judge codes, with fixes f9827c51 and 68047187 | Done (jobs 13082961, 13083896), on S3 with `REPORT.md`; judge test-retest 0.87 (kappa 0.73), 6 unreliable codes; merge rule not followed by the 30B model |
| `smoke-dais` | dais | Job-script tests with Qwen3-30B-A3B | 1-GPU test passes (job 878637); 2-GPU test (job 878812) cancelled unrun, the DeepSeek boot tested tensor-parallel 2 under the production container options; on S3 |
| `dais0` | dais | First boot of DeepSeek V4.1 Flash plus a trial on 6 versions | First boot done (job 878977, 2 B200, TP 2): server ready in 22 min, preflight, probe and stage A on 6 versions all passed; on S3. Second boot (job 879378) also 22 min: the autotune reruns despite the cache; probe ok |
| `prod1` | dais | Production run, DeepSeek-V4.1-Flash-1017 on 2 B200 (TP 2, `--max-model-len 65536`, client concurrency 128), code at commit 95e242d8 | Stage A done (job 879637: 25 min queue, 17 min load, 16 min serving, 0 failed calls), on S3 with `REPORT_stageA.md`. Codebooks v1: what 42 + 8 never implemented (9 families), why 112 (8 families), basis 27 (10 families). Next: the scientific lead reviews the codebooks; then stage B, expert check and join |
| `prod2` | dais | Codebook steps of stage A rerun on prod1's open pass; reviewed v2 codebooks (what 25, why 40, basis 16, never implemented 6) | Done through `analyse` with the final code (jobs 880491, 880688, 880989; commit 7a3876b6): never-implemented codes judged (none found in any rule), judge kappa 0.977, raters vs judge 0.99, rater vs rater 0.999; what ideas 0/1/2/3 codes 10.5/71.8/11.4/6.3%; unassigned what 10.5%, why 1.6%, basis 9.9%. Cached offline rerun byte-identical (131 files), kill-and-resume identical. On S3 with `REPORT_stageB.md` |

## Next steps

viper1 to viper4 are done (reports in `runs/annotation/viper{1,3,4}/REPORT.md`, also on S3). Tuning on the small model (Qwen3-30B-A3B on viper) is finished: the remaining errors (merges, judge order effects on overlapping codes, relation errors) are the small model not following instructions and are to be re-measured on the production model before any further prompt work.

Step 7 `analyse` is built and tested (fixture run and viper4): tables, figures, `summary.json` and a generated `REPORT.md` under `analysis/`. It needs `pandas` and `matplotlib` in `.venv-annot` (Mac only).


Notes on prod2: its raters filled the shorter sheet format 1 (codes with definitions only, no families, examples, game setting or sweep-range instruction); sheet format 2 shows all of that. Its what codebook was built before `held_out.exclude_notes_from_codebook`, so it saw the held-out versions' notes (never their rules). Its open pass (corpus, free codes, notes records, cache) was copied from prod1; the manifests say so.
1. Production run on dais (`prod1` open pass, `prod2` codebooks and stage B) is complete through `analyse`. Open issues are in `runs/annotation/prod2/REPORT_stageB.md`: a stable judge miss on `near_standard_nudge` (pf-20261105-3-v3), the what assigner still abstaining on some fitting ideas, the switch-cycle code boundary, anecdotal idea values, and an expert check by two Claude raters on the shorter sheet format.
2. Results into new tabs of the design page.

## Decisions taken

- Production model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Development model on viper: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Free-coding sample: one version per lineage (36). Expert-check sample: 30 of the remaining versions. Both are config settings chosen to get the first run going.
- The first run has no stops: codebook review and expert check are done by Claude in the same file formats a person would use. A second run uses real experts.
