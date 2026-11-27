# Sims on DAIS CPUs against Raven GPUs: are the results the same?

For `docs/plans/policy-finder-open-weights.md`. The policy finder's sims are to run on DAIS CPUs in the existing sim image `algoinst.sif` (torch 1.11; Raven's image, byte-identical), as single-threaded processes on slices of a sweep: the shape `docs/reports/dais_feasibility.md` measured as the fastest. This report tests whether that gives the same results as the Raven GPUs the pipeline uses today. Scripts: `infra/dais/gpusim_*` (the name is historical: the branch started as a GPU sim image, now a closed side branch, see the end).

Status 2026-11-27: the DAIS side, the null comparison, the slicing invariance and the deterministic checks are done; the two Raven jobs of the design are queued (Raven's GPU partition is full) and their comparison is pending.

## Summary

- **Slicing does not change results, exactly.** The same config run as 6 one-point slices, as 2 halves and whole gives bitwise identical per-episode results (36,000 of 36,000 episodes), and a real 256-point sweep split by `generate_sim_config.py --n-parts 32` and `--n-parts 16` gives identical `sweep.json` scores for its first 16 points. A rerun on another job reproduces them bitwise. Seeding is per batch, so a slice replays exactly the games the whole sim gives it.
- **Thread count barely matters:** the same config single-threaded and with 64 threads (one process) gives identical results in 35,999 of 36,000 episodes; one episode diverged (pool 0.47 apart: a different reduction order changed one sampled decision). The per-point means agree to 4 decimals. Single-threaded slices are also about 17 times faster per CPU here: 47 s per 6000-episode slice on 1 thread against 805 s for 36,000 episodes on 64 threads.
- **The null comparison is well calibrated.** Two seed sets on DAIS CPUs (12,000 against 12,000 episodes per point) differ by +0.08 +- 0.56 (shift) and chi-squared 4.4 on 5 df (contrasts) for `final1`, -1.08 +- 0.71 and 3.7 on 7 df for `v7slice`.
- **Against the existing Raven result** (`pf-20261104-1` final run, 2000 episodes per point, A100): the DAIS means at 24,000 episodes per point lie within 2 standard errors of all six Raven values, chi-squared 8.6 on 6 df (p 0.20), mean z +0.19. No sign of a difference, at Raven's resolution of about 1 point per point.
- **Raven comparison at full power: pending.** Two Raven jobs (12,000 episodes per point for `final1`) are queued, starts estimated 2026-11-28 00:28 and 15:07. With them, a per-point difference of 1.4 points, a common shift of 1.4 and a ranking contrast of 0.66 points are detectable with 80 % power.
- **Speed:** 288,000 episodes (8 configs, 56 slices) in 89 s on 48 CPUs of a `small` node, one thread per slice.
- **Models across torch and scikit-learn versions (CPU):** the AH models' outputs under torch 1.11 and 2.5 agree to 9e-7 (float32 precision); the linear punisher's probabilities are bitwise identical under scikit-learn 1.5.2 (the image) and 1.6.1 (the pickle's version), despite the `InconsistentVersionWarning`.

## Design (written before the runs; platforms revised on 2026-11-27, see below)

**Question.** Does a sim give the same distribution of results in `algoinst.sif` on DAIS CPUs, run as single-threaded slices, as in the same image on a Raven A100? Random streams differ between CPU and CUDA, so single values cannot match; the test compares distributions.

**Platforms.**

| Label | Image | Device and shape | Where |
|---|---|---|---|
| `raven_gpu` | `algoinst.sif` | A100, one process per config (as the pipeline runs a sim) | Raven, `cluster_proxy.py` (snapshot of this branch at `00df782d`, `submit`) |
| `old_cpu1` | `algoinst.sif` | DAIS CPUs, one single-threaded process per point (the production shape) | DAIS `small`, `gpusim_cpu.slurm.sh equiv` |
| `old_cpu64` | `algoinst.sif` | DAIS CPUs, one process with 64 threads per config | DAIS `small`, `gpusim_equiv.slurm.sh`; for the thread check |

All run the same code, the same model files (SHA-256 checked on Raven, DAIS and the Hetzner host: `ada5d2aa...` contribution, `f9a19012...` valid, `8a4ae4ad...` switch, `e7713cb0...` punisher bundle) and the same configs (`infra/dais/gpusim_configs/`, written by `gpusim_equiv.py configs`).

**Reference configs.**

- `final1`: the 6-point final run of `pf-20261104-1` (its GP peak and the 5 best sweep points, against `ah`), whose Raven result exists (2000 episodes, seed 1000). The 6 points lie within 5 pool points of each other: the regime where rules are ranked.
- `v7slice`: a different rule, 8 points of the sweep of `pf-20261104-12-v7` at the 0, 15, 30, 45, 60, 75, 90 and 100 % quantiles of its sweep pool (64.8 to 77.4 at 500 episodes).

**Quantities.** Per point, the sweep score's per-episode pool (definition 4 of #226, as `sweep.json` averages it) and `members` (mean size of the point's group). Compared: (1) the per-point mean; (2) the common shift (the mean over points of the per-point differences); (3) the contrasts (each point's difference minus the shift), which decide the ranking, tested jointly (chi-squared on k - 1 df); (4) the rank order.

**Common games.** Each config sets `seed_cycle`, so within a job every point plays the same games (batch b of every point seeded `seed + b`). The unit of analysis is a game (job, episode) with the vector of all points' results; covariances between points are estimated from these units, so the tests account for the correlation (observed 0.87 to 0.90 for the pool), which makes contrasts about 3 times more precise than single points.

**Seeds and size.** Per reference: 4 jobs with seeds 71000, 71010, 71020, 71030 (`final1`, 6000 episodes per point and job, batches of 2000, `seed_cycle` 3) and 72000 to 72030 (`v7slice`, 4500 per point and job, batches of 1500). Every platform runs the same seeds (the CPU and CUDA streams differ, so this pairs nothing; it only keeps the configs identical).

**Null comparison.** On each platform, the first half of its jobs against the second (DAIS: jobs 0-1 against 2-3; Raven: job 0 against job 1).

**Decision rule.** For the pool, every platform comparison gets two tests (shift: z; contrasts: chi-squared), Holm-corrected over all of them at familywise alpha 0.05. If one is rejected: "different". If none is: "equivalent within X", X being the 90 % upper bound (TOST) on the absolute shift, the largest absolute contrast and the largest absolute per-point difference. "Equivalent within 1 point" needs the shift and contrast bounds at or below 1. `members` is reported the same way, without a verdict of its own.

**Power.** Per-episode standard deviation of the pool: 46 (`final1`), 50 (`v7slice`), as planned (48). The original plan had 4 Raven jobs per reference (24,000 episodes per point, per-point MDE 1.2 points). Raven's GPU queue was full on 2026-11-27 (start estimates 7 to 21 hours out, behind the other agent's smoke job), so only 2 Raven jobs remain: `final1` jobs 0 and 1, 12,000 episodes per point; `v7slice` has no Raven run. Against DAIS's 24,000, the standard errors of a difference are: per point 46 x sqrt(1/12000 + 1/24000) = 0.51 (80 % power for 1.4 points), shift about 0.49 (1.4 points), contrasts about 0.23 (0.66 points). So the ranking is tested at better than 1 point; single-point levels and the common shift at about 1.4.

**Deterministic part.** For fixed model inputs (every AH-model call and linear-bundle call of a 64-episode sim, captured once), each image recomputes the AH models' class probabilities (`predict` with `sample=False`, RNN state carried as in the sim) and the punisher's `predict_proba` (`gpusim_forward.py`).

## Results on DAIS CPUs (`old_cpu1`, job 987769)

Pool, mean +- se per point (24,000 episodes per point for `final1`, 18,000 for `v7slice`):

| `final1` point | pool | members | | `v7slice` point | pool | members |
|---|---|---|---|---|---|---|
| gp | 65.20 +- 0.31 | 4.00 +- 0.01 | | s167 | 69.13 +- 0.40 | 4.15 +- 0.01 |
| s183 | 67.37 +- 0.31 | 4.29 +- 0.01 | | s219 | 70.48 +- 0.39 | 4.20 +- 0.01 |
| s063 | 63.80 +- 0.28 | 4.51 +- 0.01 | | s087 | 71.97 +- 0.37 | 4.41 +- 0.01 |
| s043 | 62.90 +- 0.29 | 4.52 +- 0.01 | | s002 | 70.55 +- 0.34 | 4.52 +- 0.01 |
| s255 | 65.60 +- 0.30 | 4.38 +- 0.01 | | s101 | 71.03 +- 0.35 | 4.48 +- 0.01 |
| s103 | 64.20 +- 0.29 | 4.52 +- 0.01 | | s120 | 72.01 +- 0.38 | 4.28 +- 0.01 |
| | | | | s253 | 70.58 +- 0.35 | 4.42 +- 0.01 |
| | | | | s008 | 72.95 +- 0.38 | 4.37 +- 0.01 |

Ranking: `final1` s183 > s255 > gp > s103 > s063 > s043; `v7slice` s008 > s120 > s087 > s101 > s253 > s002 > s219 > s167 (the sweep's 500-episode ranking of these 8 differs: at 500 episodes their se was 2.1 to 2.4).

**Null comparison** (jobs 0-1 against 2-3, difference = second minus first):

| | n per side | shift +- se | p | contrasts chi2 (df) | p | largest contrast +- se |
|---|---|---|---|---|---|---|
| `final1` pool | 12,000 | +0.08 +- 0.56 | 0.88 | 4.4 (5) | 0.50 | gp +0.45 +- 0.27 |
| `final1` members | 12,000 | +0.00 +- 0.02 | 0.82 | 4.2 (5) | 0.53 | gp +0.02 +- 0.01 |
| `v7slice` pool | 9,000 | -1.08 +- 0.71 | 0.13 | 3.7 (7) | 0.81 | s120 -0.29 +- 0.23 |
| `v7slice` members | 9,000 | -0.02 +- 0.02 | 0.39 | 3.5 (7) | 0.84 | s002 +0.01 +- 0.01 |

Seed sets move all points together (the shift: game draws), while the contrasts are stable to a few tenths of a point.

**Against the existing Raven result** (`pf-20261104-1` final run on Raven, its own games: seed 1000, 2000 episodes per point, no common games):

| Point | Raven A100 (2000) | DAIS CPU sliced (24,000) | difference | z |
|---|---|---|---|---|
| gp | 66.00 +- 1.07 | 65.20 +- 0.31 | -0.80 | -0.72 |
| s183 | 67.58 +- 1.07 | 67.37 +- 0.31 | -0.21 | -0.19 |
| s063 | 64.99 +- 0.98 | 63.80 +- 0.28 | -1.19 | -1.17 |
| s043 | 63.33 +- 0.98 | 62.90 +- 0.29 | -0.43 | -0.42 |
| s255 | 63.63 +- 1.03 | 65.60 +- 0.30 | +1.97 | +1.84 |
| s103 | 62.38 +- 0.98 | 64.20 +- 0.29 | +1.82 | +1.78 |

Chi-squared 8.6 on 6 df (p 0.20), mean z +0.19. Raven's points are independent of each other here (no common games), DAIS's points are correlated, which this simple test ignores; the full test waits for the Raven jobs below.

## Slicing and threads (job 987770; job 987567)

| Comparison (old image, CPU) | Result |
|---|---|
| `final1_j0` as 6 one-point slices vs 2 three-point halves vs whole, single-threaded | identical: 0 of 36,000 episodes differ, pool and members |
| The same slices in two jobs (987769, 987770) | identical |
| A real sweep of `pf-20261104-12-v7` (256 x 500), first 16 points as parts 1-2 of `--n-parts 32` vs part 1 of `--n-parts 16` | identical `sweep.json` pool and members for all 16 |
| Single-threaded slices vs one 64-thread process (`final1_j0`, job 987567) | 1 of 36,000 episodes differs (s255, pool by 0.47); all per-point means identical to 4 decimals |

The pipeline's parts are seeded by global batch index (`seed` plus the batches of the parts before), so as long as part boundaries fall on batch boundaries (500 episodes per point, batches of 1000: an even number of points per part) a slice plays exactly the games of the unsliced sweep. An odd number of points per part would shift the batch boundaries and change the games (not the distribution).

### Threads

Threads change the order of floating-point reductions, which can flip a sampled decision that lies on a probability boundary; from there that episode's game diverges. Once in 36,000 episodes here. This is not a platform difference that could bias results: an equally likely flip in either direction, invisible in the means (s255: 65.5382 both). The 64-thread run of `final1_j1` was cancelled after `j0` answered the question.

## Deterministic forward pass (job 987679, H200 node; capture from the Hetzner host)

Same inputs (74 AH-model calls, 24 punisher calls, 12,800 rows each), replayed in each image:

| Compared | contribution | valid | switch | linear punisher |
|---|---|---|---|---|
| torch 1.11 CPU vs torch 2.5 CPU | max 9.2e-7, argmax same | identical | max 1.8e-7 | identical (sklearn 1.5.2 vs 1.6.1) |
| torch 2.5 CPU vs torch 2.5 H200 | max 6.5e-4, argmax differs in 0.023 % of rows | identical | max 1.6e-4 | identical |

On the CPU the two torch versions agree to float32 precision, so neither the torch upgrade nor the scikit-learn pickle warning changes the models. On the GPU, cuDNN's RNN kernels (TF32 on Hopper) move probabilities by up to 7e-4; with sampling, this changes the random stream, not the distribution.

## Raven jobs still to come

| Raven job | Config | Purpose | Estimated start (2026-11-27 18:10) |
|---|---|---|---|
| 32997490 | `final1_j0` | `raven_gpu`, seed set 1 | 2026-11-28 00:28 |
| 32997532 | `final1_j1` | `raven_gpu`, seed set 2 | 2026-11-28 15:07 |

Snapshot `00df782def908d3b674082db4920e95802971568`; outputs on Raven in the snapshot's output tree, `plots/simulation/gpusim/final1_j{0,1}/`. To finish: on the Hetzner host, `python infra/dais/gpusim_raven.py 00df782d <out> --via-log` in a container on the command server's network (resumes from `<out>/state.json`; it reduces each job to per-episode scores on the login node and reads them back through the run log, without `sync_from_cluster`), then `gpusim_equiv.py analyze` over `raven_gpu/` and `old_cpu1/`.

The forward pass also ran on a Raven login node (torch 1.11, `capture_old.pkl`, `replay_old_cpu_raven.npz` in the snapshot's output tree); it was not brought back, since DAIS's byte-identical image covers that comparison.

## Closed side branch: a torch 2.5 GPU sim image

Started because `algoinst.sif` has no H200 kernels; dropped on 2026-11-27 when single-threaded CPU sims proved fast enough (user's decision). What exists:

- `infra/claude_proxy/container/CC_env/Dockerfile.gpu-sim`: `Dockerfile.gpu`'s Python environment (Python 3.10, torch 2.5.1+cu124, PyG wheels for torch 2.5.0+cu124, `requirements.local.txt` under `constraints.local-gpu.txt`) without the agent's tools, on `nvidia/cuda:12.8.1-base-ubuntu24.04`. Built on the Hetzner host: `docker build -f container/CC_env/Dockerfile.gpu-sim -t algoinst_gpusim:torch2.5.1-cu124 .` from `infra/claude_proxy`, image `sha256:f3bb6a8d47ff...`, 10.4 GB; `docker save | gzip -1`, 3.50 GB (`988c6e8d...`), streamed to DAIS, converted with `gpusim_build.slurm.sh` (215 s on 16 CPUs; `docker-archive:` needs the tar decompressed).
- The unused SIF: `/dais/fs/scratch/levinb/policy-finder-gpusim/images/algoinst_gpusim.sif`, 3.48 GB, SHA-256 `a19af09a6b74b9065b447a781d09678aef7b704a12814378b4d97952a06a0601`. Versions: torch 2.5.1+cu124 (cuDNN 9.1), arch `sm_50 ... sm_90`, torch_geometric 2.6.1, torch_scatter 2.1.2, torch_sparse 0.6.18, scikit-learn 1.6.1, numpy 1.26.4, pandas 2.2.3, scipy 1.14.1.
- On one H200 (job 987679): CUDA, PyG and the models work; the 20-episode smoke sweep ran through `python -m aimanager simulate` (43 s, 763 MiB of GPU memory). No equivalence, speed or memory runs were made with it.
- `validate-rule` in it works (Hetzner, CPU): `pf-20261104-1` and `pf-20261104-12-v7` pass in about 3 s. Had the sim image become torch 2.5, `image_check` would have had to run in that image, since it exists to catch code the sim image's torch rejects; with sims staying in `algoinst.sif`, `image_check` stays as it is.

### B200 support: what it takes

- The lowest official PyTorch wheels with Blackwell kernels are torch 2.7.0+cu128. Installed in a throwaway venv, it reports arch flags `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120`: H200 (sm_90) and B200 (sm_100) both covered. The DAIS B200 nodes (`gpu:b200:8`, feature `blackwell`) run driver 580.196.07, enough for CUDA 12.8.
- data.pyg.org has cp310 wheels for `torch-2.7.0+cu128` (torch_scatter 2.1.2, torch_sparse 0.6.18, torch_cluster 1.6.3, torch_spline_conv 1.2.2, pyg_lib), also for 2.8.0+cu128/cu129.
- With `torch==2.7.0+cu128` as constraint, the same install commands as `Dockerfile.gpu` resolved torch-geometric 2.6.1 and all of `requirements.local.txt`: numpy 1.26.4, scikit-learn 1.6.1 (the punisher bundle's version), pandas 2.2.3, scipy 1.14.1.
- To move the one modern environment: the torch pin and index URL in `constraints.local-cpu.txt`, `constraints.local-gpu.txt`, `Dockerfile`, `Dockerfile.gpu` (and `Dockerfile.gpu-sim`), and the PyG find-links URL. `Dockerfile.gpu` and the compose service `claude-gpu` are the colleague's RTX 4090 setup (cu128 supports the 4090, but it is his image to rebuild).
- With the sims on CPUs the GPU type no longer matters for the sims; this is only relevant if GPU sims return.
