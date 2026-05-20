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

## Joint `(prev_c, prev_p)` coverage

`uv run python experiment_analysis/joint_coverage.py` derives per-player
lag-1 `prev_contribution` and `prev_punishment`, then buckets both and
prints the joint distribution.

### Joint shares (% of all rows)

| prev_c \ prev_p | p=0 (Leg) | p=0 (50ep) | Δ |
|---|---:|---:|---:|
| 0       | 3.15  | 7.85  | +4.70 |
| 1-5     | 3.88  | 12.02 | +8.14 |
| 6-10    | 5.81  | 18.79 | +12.99 |
| 11-15   | 7.60  | 14.91 | +7.32 |
| 16-19   | 6.83  | 4.27  | −2.55 |
| **20**  | **31.24** | **12.33** | **−18.91** |

**Read.** In legacy data, **31% of all rows are `(c=20, p=0)`** — the
cooperative steady state where perfect cooperators are left
unpunished. In 50ep that cell collapses to 12%; the freed mass
migrates down to `(c=6-10, p=0)` (+13pp) and `(c=11-15, p=0)` (+7pp).
The 50ep `p=0` rows live overwhelmingly at *mid-range* contribution,
not at the cooperative ceiling.

This is consistent with PR #96's finding that the 50ep AH starts at
~8.5 (matching first-round mean 8.75) and stays there under sustained
`p=0`. The model is faithful: at `(prev_c≈8, prev_p=0)` the empirical
conditional behaviour in training is "stay around 8".

## Manager-as-confound — conditional `P(prev_p | prev_c)`

The joint distribution above mixes the contribution prior with the
manager's response policy. Conditioning on `prev_c` isolates the
policy.

### Legacy manager (% of `prev_c` row)

| prev_c | p=0 | p=1-3 | p=4-7 | p=8-15 | p=16+ |
|---|---:|---:|---:|---:|---:|
| 0     | 44 | 12 |  7 | 14 | **24** |
| 1-5   | 31 | 14 | 17 | 20 | 17 |
| 6-10  | 34 | 16 | 21 | 24 |  4 |
| 11-15 | 38 | 30 | 22 |  9 |  1 |
| 16-19 | 60 | 31 |  8 |  1 |  0 |
| 20    | **98** |  1 |  0 |  1 |  0 |

### 50ep manager (% of `prev_c` row)

| prev_c | p=0 | p=1-3 | p=4-7 | p=8-15 | p=16+ |
|---|---:|---:|---:|---:|---:|
| 0     | **66** | 10 |  7 |  9 |  8 |
| 1-5   | 56 | 19 | 13 |  7 |  4 |
| 6-10  | 65 | 20 |  9 |  5 |  1 |
| 11-15 | 76 | 15 |  5 |  3 |  1 |
| 16-19 | 79 | 15 |  3 |  2 |  1 |
| 20    | 96 |  1 |  1 |  1 |  0 |

**Read.** Both managers reward `c=20` with virtually no punishment
(96–98%) — that's the only cell where the policies agree.

- **50ep managers are dramatically more lenient on free-riders.** At
  `prev_c=0`, heavy punishment (`p=16+`) drops from 24% (legacy) to
  8% (50ep). Choosing `p=0` for the same free-rider jumps from 44% to
  **66%**.
- **Conditional elasticity weakened.** Legacy
  `P(p=0 | c=20) − P(p=0 | c=0)` ≈ 98% − 44% = **54pp gradient** — a
  sharp manager response. 50ep is 96% − 66% = **30pp** — markedly
  flatter. The 50ep human managers themselves were less elastic to
  prior contribution.

The contribution AH can't learn punishment-elasticity that wasn't
demonstrated to it in training. PR #96's no-decay-at-`p=0` and
wrong-direction-at-`p=5,10` both follow from this: the model is
faithfully reproducing a regime where punishment doesn't correlate
strongly with contribution shifts.

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

Three of five remaining angles are now closed:

1. ~~Punishment regime coverage (marginal `p` histogram).~~ — **done.**
   `p=0` is more common in 50ep than legacy (70.7% vs 59.6%); the
   higher-`p` regimes (`p≥8`) are roughly half as frequent. Coverage
   isn't the gap.
2. ~~`(prev_c, prev_p)` joint coverage.~~ — **done.** The `(c=20, p=0)`
   cooperative-steady-state cell collapses from 31% (legacy) to 12%
   (50ep). Mass migrates to mid-`c` + `p=0`, exactly the regime the AH
   reproduces in PR #96.
3. ~~Manager-as-confound.~~ — **done.** 50ep managers were dramatically
   less elastic: `P(p=16+ | c=0)` 24% → 8%; conditional gradient
   `P(p=0 | c=20) − P(p=0 | c=0)` 54pp → 30pp. The training data
   simply doesn't demonstrate strong contribution-conditional
   punishment to the model.
4. Empirical decay-under-low-`p` — find sequences in 50ep with
   sustained low `p` and measure Δc. Would partition data-vs-model
   conclusively, but the combined story from (1)–(3) is already
   strong: the data lacks both the cooperative-ceiling anchor and the
   free-rider deterrent the legacy AHs learned from.
5. Switching-effect isolation — within-block vs post-switch rounds.
   Less urgent given the manager-policy explanation; worth checking if
   the policy weakness is concentrated post-switch.

## Verdict so far

The 50ep contribution AH's failure modes in PR #96 (no decay at `p=0`,
wrong-direction elasticity at `p=5,10`) trace back to two structural
properties of the training data:

- **Cooperative ceiling missing.** Legacy data anchors on the `(c=20,
  p=0)` cell (31% of rows). 50ep has 12%. There's no dense
  "stay-at-top" steady state for the model to lean on under sustained
  no-punishment.
- **Manager elasticity weakened.** 50ep human managers were ~2× more
  lenient on free-riders and showed half the conditional-policy
  gradient. Punishment-elasticity wasn't demonstrated in training.

Retraining the contribution AH on the 50ep data alone won't fix this —
the signal isn't there. Candidate next steps: (a) collect additional
data with stricter human-manager policies, (b) mix legacy + 50ep
during training, (c) constrain the AH architecture to enforce
monotone response to `prev_punishment`.
