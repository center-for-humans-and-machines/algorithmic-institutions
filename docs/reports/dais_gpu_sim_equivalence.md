# A GPU sim image for DAIS: does it run on H200, and are its results the Raven image's?

For `docs/plans/policy-finder-open-weights.md` (decision 1: sims on the DAIS GPUs next to the model server). The Raven sim image `algoinst.sif` (torch 1.11+cu113) has no kernels for H200 (`docs/reports/dais_feasibility.md`, section 2). This report builds a torch 2.5 image from the repo's existing GPU recipe, runs the repo's sims in it on an H200, and tests whether its results are equivalent to the Raven image's. Scripts: `infra/dais/gpusim_*`.

## Design of the equivalence test (written before the runs)

**Question.** Does a sim give the same distribution of results in the new image on an H200 as in the Raven image on an A100 (the reference) and in the Raven image on DAIS CPUs? Random streams differ across devices and torch versions, so single values cannot match; the test compares distributions.

**Platforms.**

| Label | Image | Device | Where |
|---|---|---|---|
| `raven_gpu` | `algoinst.sif` (torch 1.11.0+cu113) | A100 | Raven, through `cluster_proxy.py` (snapshot of this branch, `submit`, `fetch`) |
| `new_gpu` | `algoinst_gpusim.sif` (torch 2.5.1+cu124) | H200 | DAIS `gpu1` |
| `old_cpu` | `algoinst.sif` | CPU | DAIS `small` |
| `new_cpu` | `algoinst_gpusim.sif` | CPU | DAIS `small`; separates the torch version from the device |

All four run the same commit, the same model files and the same configs (`infra/dais/gpusim_configs/`, written by `gpusim_equiv.py configs`).

**Reference configs.**

- `final1`: the 6-point final run of `pf-20261104-1` (its GP peak and the 5 best sweep points, against `ah`), whose Raven result exists (2000 episodes, seed 1000). The 6 points lie within 5 pool points of each other: the regime where rules are ranked.
- `v7slice`: a different rule, 8 points of the sweep of `pf-20261104-12-v7` at the 0, 15, 30, 45, 60, 75, 90 and 100 % quantiles of its sweep pool (64.8 to 77.4 at 500 episodes): different punishing behaviour and a wider spread.

**Quantities.** Per point, the sweep score's per-episode pool (definition 4 of #226, as `sweep.json` averages it) and `members` (mean size of the point's group). Compared: (1) the per-point mean; (2) the common shift (the mean over points of the per-point differences between platforms); (3) the contrasts (each point's difference minus the shift), which decide the ranking, tested jointly (chi-squared on k - 1 df); (4) the rank order of the points.

**Common games.** Each config sets `seed_cycle`, so within a job every point plays the same games (batch b of every point seeded `seed + b`). Points' per-episode results are then positively correlated, which makes contrasts more precise than the per-point means. The unit of analysis is a game (job, episode), with the vector of all points' results; covariances between points are estimated from these units, so the tests account for the correlation.

**Seeds and size.** Per reference and platform: 4 jobs with seeds 71000, 71010, 71020, 71030 (`final1`, 6000 episodes per point and job, batches of 2000, `seed_cycle` 3) and 72000 to 72030 (`v7slice`, 4500 episodes per point and job, batches of 1500). Every platform runs the same seeds. Jobs 0 and 1 form seed set S1, jobs 2 and 3 set S2. Per point and platform: 24,000 (`final1`) and 18,000 (`v7slice`) episodes; 36,000 episodes per job.

**Power.** The observed per-episode standard deviation of the pool is about 48 (`pf-20261104-1` final: se 1.07 at 2000 episodes; `v7` sweep: about 49). For a per-point difference between two platforms, se = 48 x sqrt(2/24000) = 0.44 (`final1`) and 0.52 (`v7slice`); the difference detectable with 80 % power at two-sided alpha 0.05 is 2.8 se = 1.2 and 1.5 points. The shift averages over points and the contrasts profit from the common games, so both are more precise than a single point; their actual standard errors are reported. Detecting 1 point per point with 80 % power would need 2 x (2.8 x 48)^2 = 36,000 episodes per point and platform, 1.5 times this budget; the design accepts 1.2 to 1.5 points per point and aims at 1 point or better for the shift and the contrasts.

**Null comparison.** On each platform, S2 against S1: two seed sets, same image and device. Its shift, contrast chi-squared and per-point differences show the seed-to-seed spread at half the sample size; a platform comparison should look like it.

**Decision rule.** For the pool, every platform comparison gets two tests (shift: z; contrasts: chi-squared), Holm-corrected over all of them at familywise alpha 0.05. If any comparison with `new_gpu` is rejected, the verdict is "different". If none is, the verdict is "equivalent within X", where X is the 90 % upper bound (TOST) on the absolute shift, on the largest absolute contrast and on the largest absolute per-point difference between `new_gpu` and `raven_gpu` (and `old_cpu`); "equivalent within 1 point" needs the shift and contrast bounds at or below 1. If the bounds exceed what the budget can resolve, the verdict is "not decidable with this budget" with the bounds reached. `members` is reported the same way, without a verdict of its own.

**Deterministic part.** For a fixed set of model inputs (the inputs of every AH model call and every linear-bundle call in a 64-episode sim, captured once), each image recomputes on the CPU the AH models' class probabilities (`predict` with `sample=False`, the RNN state carried as in the sim) and the linear punisher's `predict_proba` (`gpusim_forward.py`). Torch 1.11 against 2.5 should agree to float32 precision (about 1e-6); any larger difference is explained first.
