# Step classification: status and how to resume

Last updated: 2026-11-13, 00:40. This file is the handover for picking the work up on another machine or in another session. Update it whenever the state changes.

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

## State

The pipeline is built, reviewed twice and has run end to end on the production model (`prod2`). A forced rerun with no server reproduces every output byte for byte from the stored replies, and a killed step resumes to the same result. 39 tests pass. Results and figures are in the design page (tabs Production run and Results); the page source and its figures are in `docs/step_classification/`.

Small-model tuning on viper (`viper1` to `viper4`) is finished; viper stays useful for testing code changes cheaply.

Notes on `prod2`:

- Its codebook review and its two expert-check raters were Claude, not field experts. The raters agree with each other on 749 of 750 answers, which is not a human baseline.
- Its raters filled sheet format 1 (codes with definitions only). Format 2 also shows families, examples, the game setting and the sweep-range instruction.
- Its what codebook was built before `held_out.exclude_notes_from_codebook`, so it saw the held-out versions' notes, never their rules.
- Its open pass comes from `prod1`; the run's manifests record that.

Known weaknesses, detailed in `runs/annotation/prod2/REPORT_stageB.md` (also on S3): 10.5% of notes ideas get no what code, about half of them wrongly; the judge consistently misses `near_standard_nudge` in `pf-20261105-3-v3`; the two switch-cycle codes have the lowest reliability (kappa 0.82 and 0.83); the score effect of an idea cannot be estimated from this corpus.

## Next steps

1. Second production run with field experts. Start a fresh run id so the held-out notes are excluded from codebook building: `corpus`, stage A on dais, then hand `codebook/*.v1.yml` and `review begin` to the reviewers, run the literature step on the reviewed codebooks, stage B on dais, `expert-check export` (sheet format 2) to two raters, `expert-check ingest`, `join`, `analyse`.
2. Put the reviewer's 12 open questions (`runs/annotation/prod2/review/comments.md`) to the experts before or during that review.
3. Have a person check the 11 references the literature step could not confirm (`runs/annotation/prod2/review/literature_notes.md`).
4. Optional: a second judge model (`zai-org/GLM-5.3` needs 5 B200, so the exclusive partition) to report how much the judge model matters.

## Decisions taken

- Production model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Development model on viper: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Free-coding sample: one version per lineage (36). Expert-check sample: 30 of the remaining versions. Both are config settings chosen to get the first run going.
- The first run has no stops: codebook review and expert check are done by Claude in the same file formats a person would use. A second run uses real experts.
