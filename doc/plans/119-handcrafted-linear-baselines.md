# [DRAFT] Hand-crafted linear baselines for contribution & switch (issue #119)

## Goal

Build the strongest **linear** artificial-human baseline the GNN must beat, for both the
contribution and switch targets. The existing minimal baselines
(`scripts/baselines/{contribution,switch_logit}_baseline.py`) test one or two fixed feature
sets; here we search a structured pool of hand-crafted features to find the best linear model,
then report an honest, un-overfit number against a locked holdout.

Reference bars (both from `reports/ah_baseline_log_loss.md`):
- Contribution GNN test log-loss = **1.9897**; constant floor = **2.8787**.
- Switch GNN test log-loss = **0.5163**; constant floor = **0.6095**.

The search is a **block-level grid** exploiting the measure×frame factorial structure of the
feature pool, run under **nested CV with a locked holdout** and **1-SE selection** — not an
exhaustive subset search and not regularization-only pruning. This plan fully supersedes the
root scratch file `feature_blocks.md`, which is to be **deleted**.

Decisions in this plan are settled (they answer the issue's four open questions); do not
re-litigate them:
- **CV granularity:** pair/episode level via `get_cross_validations(group_key=pair_id)`,
  seed 38381. Training rows are per-(episode, agent, round); the split stays pair-level because
  lagged features + the pair-flip doubling would leak otherwise.
- **Model:** linear only — multinomial `LogisticRegression` (categorical target) and `Ridge`
  (continuous target). Tune regularization only (`C` / `alpha`). No trees / XGB.
- **Contribution target:** produce BOTH categorical (21-level log-loss, comparable to GNN) and
  continuous (RMSE/MAE, not log-loss comparable). Switch target is categorical only.
- **No `src/` changes:** only `create_torch_data` and `get_cross_validations` are imported;
  every derived feature, the holdout wrapper, the nested-CV loop and the 1-SE rule live in the
  baseline script.
- **Runs locally** (CPU torch, no PyG). Per CLAUDE.md the *test suite* must run on Raven, but
  these tiny linear fits run on the local machine.

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Shared module `handcrafted_grid.py` | Feature engineering + block grid + nested-CV + 1-SE harness | |
| 2 | Contribution entrypoint | Thin wrapper reading the existing contribution config | |
| 3 | Switch entrypoint | Thin wrapper reading a new switch config | |
| 4 | Switch config | New YAML mirroring the contribution config (categorical only) | |
| 5 | Feature-computation spec | Concrete recipe for all 30 features / 24 derived | |
| 6 | Report | Results doc in `reports/` matching existing style | |
| 7 | Delete `feature_blocks.md` | Remove the root scratch file (absorbed here) | |

### 1. Shared module `scripts/baselines/handcrafted_grid.py`

Single library holding all logic so the two entrypoints stay thin; do NOT edit the existing
minimal baselines.
- **Config loader:** parse `configs/training/baselines/{contribution,switch}/*.yml` — the
  contribution grid config (`handcrafted_grid.yml`) is the **source of truth** for blocks,
  encodings, regularization, targets, CV seed/folds/holdout, `comparison_triples`, and the
  data/target/mask fields.
- **Feature engineering:** builds the full 30-feature pool from the `create_torch_data` tensors
  (see §5). Compute once per fold's train/test data dict, then select columns per block config.
- **Grid enumeration:** expand `comparison_triples` (5) × B1 × B5 × B6 × B7 (2⁴) = 80 block
  configs × 2 encodings × regularization grid × target(s). Apply the group-size **dedup** (§5).
- **Nested-CV + holdout + 1-SE harness** (§ below).
- **Metrics:** categorical → `log_loss(labels=range(N_LEVELS))` reusing the `full_proba`
  pattern from `contribution_baseline.py` (train folds may miss classes); continuous → RMSE +
  MAE. Standardize features with `StandardScaler` fit on the inner-train fold only.
- WHY one module: the harness (nested CV, 1-SE, dedup, feature engineering) is identical across
  targets; only config + metric differ.

### 2. Contribution entrypoint `scripts/baselines/contribution_handcrafted.py`

- Thin `main()` that loads `configs/training/baselines/contribution/handcrafted_grid.yml`,
  calls the shared harness, prints the summary, and writes/updates the report table.
- Runs BOTH `categorical` and `continuous` targets (per `targets: [categorical, continuous]`).
- WHY separate from `contribution_baseline.py`: that file is the minimal apples-to-apples GNN
  comparison and stays untouched as documentation of the plain baseline.

### 3. Switch entrypoint `scripts/baselines/switch_handcrafted.py`

- Same shape as §2 but loads the switch config; categorical only.
- Must pass `switch_every=SWITCH_EVERY` (4) to `create_torch_data` so `does_switch` /
  `switch_valid` land on decision rounds (see `switch_logit_baseline.py`).
- WHY: switch has no continuous variant; `does_switch` is binary.

### 4. Switch config `configs/training/baselines/switch/handcrafted_grid.yml` (NOT created here)

Structure mirrors the contribution config; the engineer creates it:
- `target: does_switch`, `mask: switch_valid`, `targets: [categorical]`.
- Same `cv` (seed 38381, `n_folds: 5`, `holdout_fold: 0`, `select_rule: one_std_error`),
  same `encodings`, same `blocks`, same `comparison_triples`.
- Regularization: `C` grid only (no `alpha`, since no continuous target).
- Data: `experiments/2group_8agent_50ep.csv`, `experiment_names: [ah_group_switching]`, plus a
  `switch_every: 4` field the entrypoint reads.
- WHY: keeps both targets config-driven and consistent; only the target/mask/targets differ.

### 5. Feature-computation spec (in `handcrafted_grid.py`)

The 30-feature pool (names exactly as in `handcrafted_grid.yml`). ~6 come straight from
tensors; ~24 are derived but collapse to **4 real helpers**. All behavioural features are
previous-round (t−1); B7 is structural (known at t). WHERE: helper functions in the module.

- **Direct from tensors** (`create_torch_data`): `prev_contribution`, `prev_punishment`,
  `prev_common_good`, `round_number`, `does_switch`→`prev_does_switch` (via existing `shift`),
  `agent_group`/`prev_agent_group` (for grouping only, not a feature). `prev_contribution_mean_peers`
  can reuse the existing `own_grp_prev_mean_contr` tensor.
- **Helper 1 — group means** (generalize `group_prev_means` from `contribution_baseline.py`):
  compute leave-one-out own-group mean and other-group mean for **each** of {contribution,
  punishment, common_good}. Yields `prev_{measure}_mean_peers` and `prev_{measure}_mean_other`.
  Reuse the exact LOO / other-group logic already in that helper; parameterize by measure.
  Note: `prev_common_good_mean_peers` is omitted from B2 by design (group-level cg equals B1's
  `prev_common_good`).
- **Helper 2 — gaps:** `prev_{measure}_mean_gap = peers_mean − other_mean` for each measure
  (B4), and the group-size deltas below.
- **Helper 3 — payoff arithmetic:** `payoff = 20 − contribution − punishment + common_good`
  (per `reports/basics.md` rules; cg is already per-capita). Apply elementwise to produce
  `prev_payoff` (self, B1 compact) and to the group-mean / gap / window aggregates
  (`prev_payoff_mean_peers`, `prev_payoff_mean_other`, `prev_payoff_mean_gap`,
  `win_payoff_mean_*`). Because payoff is linear in {c,p,cg} it is collinear with them and is
  therefore only ever used in the `compact` encoding, never alongside components.
- **Helper 4 — since-switch windows + counters:** for each agent, a running mean over the
  current tenure (rounds since the agent last changed groups) of the own-group and other-group
  measures → `win_{measure}_mean_peers` / `win_{measure}_mean_other` (B5/B6). Same helper
  derives the structural counters: `rounds_since_switch`, and `group_size` /
  `prev_group_size` / `prev_group_size_other` / `prev_group_size_delta` (per-round count of
  members in own vs other sub-group; delta = own − other). Tenure boundaries come from the
  `does_switch` / `prev_agent_group` sequence.

**Group-size dedup (must flag to engineer):** group size is sourced from **B7 in components**
runs and from **B2/B3/B4 in compact** runs, so it appears exactly once per encoding — except
when *both* B2 and B3 are on in a compact run, they each contribute a group-size feature
(`prev_group_size` and `prev_group_size_other`); those are distinct and both kept. The genuine
dedup case is: never let B7's `group_size` (components) and a B2/B3/B4 group-size feature
(compact) co-occur — guaranteed by the per-encoding block lists, but the column-assembly step
must de-duplicate by feature name defensively so a feature added by two active blocks is not
stacked twice. Also: the fully-empty config is identical across both encodings → 159 distinct
feature×encoding configs (1 duplicate), which the enumerator should collapse.

### 6. Nested-CV + holdout + 1-SE harness (in `handcrafted_grid.py`)

- **Outer split:** `get_cross_validations(data, 5, 1.0, group_key=pair_id)` produces 5
  pair-level folds. **Fold 0 = locked holdout** (config `holdout_fold: 0`); it is scored
  exactly once at the very end and never used in selection.
- **Inner selection:** over folds 1–4 (4 inner folds), for every grid cell (block config ×
  encoding × reg × target) compute mean inner-CV score and its per-fold std / SE. Reuse the
  `flatten`/`full_proba` patterns and re-fit `StandardScaler` per inner-train fold.
- **1-SE rule:** among all cells for a target, find the best mean inner score, then pick the
  **simplest** config within 1 SE of it. Define "simplest" as fewest active feature columns
  (tie-break: fewer blocks, then more regularization). WHY: guards against overfitting the grid
  and yields an interpretable model.
- **Final score:** refit the 1-SE-selected config on folds 1–4 combined and evaluate once on
  fold 0. Report that holdout number vs GNN ref and constant floor.
- WHY nested: the outer holdout gives an unbiased estimate of the selected config; the inner CV
  does model selection without touching the holdout.

### 7. Report `reports/handcrafted_baseline.md`

- Match the style of `reports/ah_baseline_log_loss.md`: header explaining data/seed/fold/mask
  identity, "lower is better", repro command.
- **Contents:** (a) inner-CV loss ranked over the grid — which blocks / encoding matter (the
  "which features help" answer); (b) the 1-SE-selected config per target and its **holdout**
  number vs GNN ref and constant floor; (c) per-block interpretation (e.g. B1 self-inertia
  dominance, B4 gap for switch); (d) contribution continuous target reported as RMSE/MAE with a
  note that it is NOT log-loss-comparable to the GNN, only the categorical is.
- Add a caveat mirroring the existing report about `random`-state reproducibility of fold
  membership.

## Implementation notes

- `create_torch_data` tensors are `[G, A, T]` (episode/pair, agent, round). All feature helpers
  operate in this shape then `flatten` with the mask (`contribution_valid` / `switch_valid`),
  exactly as the existing baselines do.
- `prev_*` features already exist for the raw tensors via `shift`; the derived group/gap/window
  features must be computed on the current-round tensors and then themselves be t−1 aligned
  (compute the aggregate at round t, then shift by one so it predicts t+1) — or equivalently
  compute directly from `prev_*` tensors as `group_prev_means` already does. Keep one convention
  and document it to avoid leakage.
- Regularization grids differ by estimator: `LogisticRegression(C=...)` vs `Ridge(alpha=...)`.
  The config already lists both; the harness picks the list matching the target.
- ~1,920 grid cells × 4 inner folds ≈ 7,680 tiny fits → seconds-to-minutes locally. If it drags,
  cache the per-fold feature pool (built once) and only re-slice columns per cell.

## Next Actions

- [ ] Add `scripts/baselines/handcrafted_grid.py` (shared feature-eng + grid + nested-CV + 1-SE).
- [ ] Add `scripts/baselines/contribution_handcrafted.py` (loads existing contribution config).
- [ ] Create `configs/training/baselines/switch/handcrafted_grid.yml` (mirror contribution,
      `does_switch` / `switch_valid`, categorical only, `switch_every: 4`).
- [ ] Add `scripts/baselines/switch_handcrafted.py` (loads switch config).
- [ ] Run both locally; write `reports/handcrafted_baseline.md` with grid table + holdout number.
- [ ] Delete root `feature_blocks.md` (content absorbed into this plan).
