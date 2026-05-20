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

## Bucket coverage — contribution and punishment

`uv run python experiment_analysis/compare_datasets.py` now also prints
bucketed shares for both feature targets.

### Contribution buckets

| bucket   | Legacy | 50ep |
|----------|-------:|-----:|
| `c=0`    | 7.5%   | 12.0% |
| `c=1-5`  | 12.3%  | 21.6% |
| `c=6-10` | 16.9%  | 28.5% |
| `c=11-15`| 19.6%  | 19.4% |
| `c=16-19`| 11.3%  | 5.4%  |
| `c=20`   | 32.5%  | 13.1% |

50ep mass is shifted left: more zero / low / mid-low contributions
(`c≤10`) accounts for 62% of rows vs 37% in legacy. Perfect
cooperation `c=20` shrinks from 32.5% to 13.1%. Players in 50ep
simply contribute less on average.

### Punishment buckets

| bucket   | Legacy | 50ep |
|----------|-------:|-----:|
| `p=0`    | 59.6%  | **70.7%** |
| `p=1-3`  | 14.7%  | 14.7% |
| `p=4-7`  | 11.4%  | 7.5%  |
| `p=8-15` | 9.5%   | 4.7%  |
| `p=16+`  | 4.8%   | 2.5%  |

**Read.** The "model never saw `p=0` in training" version of the
coverage hypothesis is dead — 70.7% of 50ep rows are at `p=0`. But the
dominance flips the question: when the punishment feature is a constant
zero on most rows, it carries no within-row variance to learn from on
most rows. Combined with PR #84's finding that `prev_contribution` is a
near-perfect predictor on its own, the loss-minimizing strategy is to
treat `prev_punishment` as low-information and rely on the
autoregressive signal.

**Caveat for PR #96 sim regimes.** `p≥8` is only ~7% of 50ep training
rows. The wrong-direction elasticity we saw at `p=10/20/30` in the
sustained-`p` sweep is largely extrapolation. `p=5` (in or near the
`p=4-7` bucket, 7.5% of rows) is closer to in-distribution and is
arguably the more trustworthy signal that the model has the elasticity
direction wrong.

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

Step 0 establishes the headline: punishment-related signal is roughly
half as strong in 50ep as in legacy, the autoregressive trap is more
entrenched, players contribute less on average, and `p=0` dominates
70% of training rows. The naive "no-deterrent regime missing" version
of the coverage hypothesis is dead, but **`p=0` imbalance** is now the
leading explanation — most rows carry no variance in the punishment
feature, so the loss-minimizing model treats it as low-information.

Issue #97 lists five more angles:

1. ~~Punishment regime coverage (marginal `p` histogram).~~ — **done.**
   Headline: `p=0` is more common in 50ep than legacy (70.7% vs 59.6%);
   the higher-`p` regimes (`p≥8`) are roughly half as frequent. Coverage
   isn't the gap; the imbalance toward `p=0` is the candidate
   mechanism.
2. `(prev_c, prev_p)` joint coverage — find the empty cells. With `c=20`
   share dropping from 32.5% to 13.1% in 50ep, the *(high-c, low-p)*
   transition cell — where free-riding-emergence would be learned — may
   be especially sparse even though both marginals are present.
3. Manager-as-confound — `p[t] | prev_c[t-1]` joint. Even with 70%
   `p=0` rows, if those rows sit predominantly at low `c` (the decay is
   already complete by then), the model never sees a high-c→p=0
   transition and can't learn "no punishment causes drop". Closely
   related to (2).
4. Empirical decay-under-low-`p` — find sequences in 50ep with
   sustained low `p` and measure Δc. Partitions data-vs-model failure
   conclusively.
5. Switching-effect isolation — within-block vs post-switch rounds.

(2) and (3) together are the highest-value next step — both probe
whether `p=0` dominance is structurally tied to low `c` (which would
make the imbalance even more pernicious than the marginal numbers
suggest).
