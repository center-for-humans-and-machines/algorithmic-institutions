# 50ep Training Data Audit — Step 0

Tracks issue #97. Builds on PR #96 (sustained-`p` sweep showed the 50ep
contribution AH has no decay at `p=0` and wrong-direction elasticity at
`p=5,10`) and PR #84 (showed `prev_contribution` dominates the model
with `prev_punishment` effectively unused).

**Question:** is the `prev_punishment → contribution` signal actually
present in the 50ep training data, or is the model faithfully reflecting
data that already lacks the signal?

## Step 0 — existing scripts

Two scripts moved from `scripts/data_analysis/` to `experiment_analysis/`
and run before any new diagnostic angles:

- `experiment_analysis/compare_datasets.py` — marginal stats across
  pilot CSVs. Extended to include `2group_8agent_50ep.csv`.
- `experiment_analysis/feature_data_correlation.py` — mutual info / RF
  importance / Spearman of training features against the target,
  computed on the raw CSV.

## Dataset-level comparison

`uv run python experiment_analysis/compare_datasets.py`:

| Metric                       | Legacy | GS 50 ep |
|------------------------------|-------:|---------:|
| Episodes                     | 135    | 50       |
| Players / episode            | 4      | 8        |
| Rounds                       | 16     | 24       |
| Mean contribution            | 13.00  | 9.18     |
| `contribution=20` share      | 32%    | 13%      |
| Contribution entropy (bits)  | 3.67   | 4.08     |
| First-round mean             | 11.29  | 8.75     |
| Last-round mean              | 14.49  | 9.18     |
| Mean common_good             | 71.42  | 67.74    |

**Read.** Legacy trajectory climbs (11.3 → 14.5) over 16 rounds with full-cooperation (`c=20`) hit on 32% of rows. The 50ep dataset stays flat at ~9 across 24 rounds and only reaches `c=20` on 13% of rows. The behavioural regime the legacy AHs learned from is structurally absent in the GS data.

**Episode-length caveat.** The protocols differ in horizon too (16 vs 24 rounds). When comparing this report's tables back to simulation outputs (e.g. PR #96), the v4 BC stack is in-distribution only through round 15 — rounds 16–23 of any sim using v4 BC AHs are extrapolation. Most of the contribution decay at `p=0` already happens in-distribution (11.5 → 5.7 by r15), so the qualitative reads hold, but quote r15 numbers when matching the legacy training horizon.

## Feature-target signal strength

`uv run python experiment_analysis/feature_data_correlation.py <config>`
for legacy (`script_21_no_grid.yml`) and 50ep
(`group_switching_contribution_50ep.yml`):

| Feature → contribution     | Metric         | Legacy  | 50ep    | Δ (×)   |
|----------------------------|----------------|--------:|--------:|--------:|
| **`prev_contribution`**    | Mutual info    | 0.6305  | 0.8028  | +27% stronger |
|                            | Spearman ρ     | +0.7333 | +0.7638 | similar |
| **`prev_punishment`**      | Mutual info    | 0.0801  | 0.0418  | −48% weaker |
|                            | RF importance  | 0.1620  | 0.1120  | −31% weaker |
|                            | Spearman ρ     | −0.3282 | −0.1676 | −49% weaker |
| RF train accuracy          |                | 0.5847  | 0.5071  | drops |

**Read.** The 50ep dataset has roughly **half the punishment signal** for
predicting next-round contribution (mutual info, Spearman both ~−50%)
and a **stronger autoregressive structure** (mutual info on
`prev_contribution` climbed 0.63 → 0.80). RF train accuracy drops
from 0.58 to 0.51, consistent with less learnable structure overall.

So the model's behaviour in PR #96 isn't an architecture failure: the
training data itself has substantially less of the punishment-elasticity
signal that the legacy AH learned, plus more "stay where you are"
structure. The model is faithfully reflecting the data.

## What's still open

Step 0 establishes the headline: signal is roughly half as strong, and
the autoregressive trap is more entrenched in 50ep. This doesn't yet
explain *why* the signal weakened. Issue #97 lists five more angles:

1. ~~Punishment regime coverage (marginal `p` histogram).~~ — partly
   addressed by `compare_datasets.py` for `contribution`; need the same
   marginals for `punishment` per dataset.
2. `(prev_c, prev_p)` joint coverage — find the empty cells.
3. Manager-as-confound — `p[t] | prev_c[t-1]` joint. If the manager's
   policy is a tight function of prev contribution, `prev_p` becomes a
   collinear copy of `prev_c` and `prev_p`'s residual signal vanishes
   in any model that already has `prev_c`. This is the most likely
   causal story for the 50% drop above.
4. Empirical decay-under-low-`p` — find sequences in 50ep with
   sustained low `p` and measure Δc. Partitions data-vs-model failure.
5. Switching-effect isolation — within-block vs post-switch rounds.

(3) is the highest-priority next step.
