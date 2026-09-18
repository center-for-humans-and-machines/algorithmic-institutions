# The punisher must see the contribution it punishes (re-baseline)

## 1. Declaration

**Slot:** punisher -- both families (the `lin_multinomial` baseline every top stack uses and the GNN punisher), plus the shared legality rule in `scripts/baselines/handcrafted_grid.py` / `src/aimanager/simulation/linear_ah.py` that enforced the lag. A shared-code bug fix in the sense of §4: fix first, then before/after scores.

**Parent:** PR #181 (`auto/contribution-punishment-response`, `[FAIL]`, at `81c8002`): it carries the group-vnode trunk (PR #179), the stimulus-skip trunk, and the copula scripts, and its RCB investigation is where the lag was found. Branch `auto/punisher-current-contribution` is created from it and merges `origin/rcb-alternative-response-slope` (`699f921`, based on `main` `b174f90`), which adds the RCE row to the evaluation suite; the RCF cell row that branch also carried is dropped again (the user chose RCE only; `reports/rcb_alternative_comparison.md` still discusses both). Isolated remote dirs: `~/repros/ai-runs/punisher-current-contr-tests` (stage A tests), stage C names its own.

**Base models.** Punisher: `artifacts/baselines/punishment_multinomial_best_with_contr.joblib` (features `prev_contribution, prev_punishment, round_number, is_first`, C = 1.0, the `handcrafted_grid_cat.yml` grid's best entry with `prev_contribution`; commit 03507c4) and its severity-copula copy `punishment_multinomial_severity_copula.joblib`; GNN punisher `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` (`x_encoding = prev_contribution, prev_punishment, is_first`). Contributor and switch slots untouched.

**Target rows:** RCE (new, protected from this branch on: §2) and RPA, the two rows that are by definition the distribution of p_t given c_t; RCC (the population of punished full contributors) and RCB (whose rate `p / (20 - c_t)` mixes a c_t denominator with a c_{t-1}-driven numerator). Every other row is a watch item, because the punisher artifact sits in every stack and this change moves them all -- which is why the ledger is re-baselined (stage D) rather than gated.

### Hypothesis

**The finding (verified, `punisher_lag_check.md`).** Both artificial punishers decide round t's punishment from round t-1's contribution, while the human manager punishes round t's. In the data, `common_good_t = 1.6 * sum(c_t) - sum(p_t)` holds in 100.0% of valid rows and `payoff_t = 20 - c_t - p_t + common_good_t / n` in 89% (57% with p_{t-1}), so the `punishment` column at (episode, t, player) is the punishment applied to round t's contribution. Behaviourally: corr(p_t, c_t) = -0.277 vs corr(p_t, c_{t-1}) = -0.189 (n = 8,431 after the eval suite's pair dedup); OLS `p_t ~ c_t + c_{t-1}` gives -0.242 on c_t and +0.067 on c_{t-1}; and P(p > 0 | c_t <= 4, c_{t-1} = 20) = 0.571 (n 49) against P(p > 0 | c_t = 20, c_{t-1} <= 4) = 0.179 (n 39). The human manager reacts to this round.

**Where the lag lived.** Training: `generic/data.py::shift` rolls every tensor by one round, so `prev_contribution[t] = contribution[t-1]`; the GNN config's `x_encoding` and the linear grid's legal set (`validate_feature_legality`, which hard-erred on any current-valued feature for the punishment target) only ever admitted the shifted tensor, and the bundle `punishment_multinomial_best_with_contr.joblib` accordingly carries `prev_contribution` ("with_contr" meant the lag). Simulation: `linear_ah.py::get_punishments` and `api_manager.py::create_data` both had round t's contribution at the last index and did not read it; the env's ordering (`update_contribution` -> `punish` -> the `prev_` copy in `step`) was already right, in the simulation loop and in RL training. Training and simulation were consistently lagged: a faithful model of the wrong mechanism, inherited from the contributor's leak rule (where `prev_` is correct) with no behavioural rationale recorded; `reports/recommended_ah_configs_50ep.md` had flagged it and recommended `contribution` (current), never implemented.

**Every sim shows it.** OLS of p on (c_t, c_{t-1}): human -0.242 / +0.067; PR 179 +0.011 / -0.083; PR 181 +0.002 / -0.081; PR 170 -0.006 / -0.079; `main` gnn x gnn x lin_multinomial +0.008 / -0.078; `main` with the GNN punisher -0.009 / -0.018. P(p > 0 | c_t = 20): human 0.038, sims 0.215-0.334. The cross-tab flips sign in the multinomial stacks (0.34 / 0.23 against the human 0.18 / 0.57).

**Behavioural rationale (one sentence, §5):** the manager punishes what a player just contributed, so the punisher must read round t's contribution -- the rows that should move are RPA, RCC, RCB and RCE.

**The change.** `contribution` (round t, the same-round input the manager has) is admitted for the punishment target in both families; `prev_contribution`, `prev_punishment`, `round_number` and `is_first` stay; `punishment`, `payoff`, `common_good` at round t and everything built from them remain illegal (they contain p_t). Both punishers are retrained with the new feature (stage C), the severity copula is re-stamped on the new multinomial bundle, the most promising stacks are re-run, and the ledger is re-baselined with RCE as the first protected row (stage D).

### Artifact naming contract (stages B-D depend on these exact names)

| what | path |
|---|---|
| baseline config | `configs/training/baselines/punishment/multinomial_current_contr.yml` (the `best_with_contr` grid entry, `contribution` added; one feature set: `contribution, prev_contribution, prev_punishment, round_number, is_first`) |
| baseline artifact | `artifacts/baselines/punishment_multinomial_current_contr.joblib` (`run_baseline_cv.py` on the config, then `inspect_best_model.py <cv csv> --config <config> --save-best --name punishment_multinomial_current_contr.joblib`; commands in the config header) |
| severity-copula copy | `artifacts/baselines/punishment_multinomial_current_contr_severity_copula.joblib` (`punishment_copula_rho.py --roundtrip --bundle <baseline artifact> --out <this path>`; the `--bundle` / `--out` flags are new on this branch, the defaults are the old paths) |
| GNN config | `configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr.yml` (`rnn_edge_50ep_doubled.yml` with `contribution` numeric first in `x_encoding`) |
| GNN artifact dir | `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr/` -- set explicitly as `output_dir` in the config (train.py takes the artifact dir from `output_dir`, not from the config's file name); the model file inside is `model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`, from the unchanged `labels` |

## 2. Plan

| # | step | stage |
|---|---|---|
| 1 | Branch from PR #181, merge the RCE branch, drop RCF, verify the eval-suite tests locally. | A (done) |
| 2 | Admit round t's contribution for the punishment target in both families: `PUNISHMENT_LEGAL_CURRENT` / `illegal_current_features` in `handcrafted_grid.py`, the load-time assert in `linear_ah.py`, `punishment_baseline.py` FEATS; document the timing in `create_data`, `_pool_from_rounds`, `notes/baseline_feature_defs.md`. No change to the env ordering (already correct) or to the contributor / switch slots. | A (done) |
| 3 | The two configs of the naming contract; `--bundle` / `--out` on `punishment_copula_rho.py` so the stamped copy can carry its name. | A (done) |
| 4 | Tests that round t's contribution reaches the punisher at round t on the linear and the GNN path, plus the legality rule (`src/aimanager/tests/test_punisher_current_contribution.py`); all tests green on Raven. | A (done) |
| 5 | Protocol: RCE as the first protected row in `notes/autoresearch.md` §2, the row count 21 -> 22, the re-baseline paragraph, §5's feature rule corrected. | A (done) |
| 6 | Train the multinomial punisher on the new config, stamp the copula, train the GNN punisher on Raven; record CV / test log-loss against the old bundle's 1.3031 (floor 1.3561) and the GNN's 1.2030. | C |
| 7 | Re-run the most promising stacks with the new punishers (23-family protocol), fetch, evaluate. | C |
| 8 | Re-baseline the ledger: new score matrix and ranking with RCE, reset the confirmed scores of the frontier PRs, record the before/after of every row for the top stack. | D |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| | (baseline) lagged punisher, PR #181 stack | | | | baseline |
| | current-contribution punishers (multinomial + copula, GNN) | | | | |

### Stage C: training and simulation

_(to be filled by stage C: log-loss of both punishers against their lagged predecessors, copula rho against 0.0395 / 0.0436, wall time, the stacks re-run and their sim dirs)_

### Stage D: re-baseline

_(to be filled by stage D: the new score matrix, the new ranking, the RCE band slopes per stack against the human +0.1397 / +0.1038 / -0.0767 / -0.1615, and the reset baselines)_

## 4. Notes

1. Stage A: the lag was a modelling error, not an indexing bug -- training and simulation were consistently lagged, so no artifact trained before this branch is affected in any way other than by conditioning on the wrong round. The fix is a legal-set change plus two configs; every code path already carried c_t at the right index.
2. Stage A: `punishment_baseline.py` (the GNN-parity diagnostic) now lists `contribution` too, so its printed baseline is comparable to the new GNN config, not the old one; its `GNN_REF` constant still quotes the lagged GNN's 1.2030 until stage C replaces it.
3. Stage A: RCF was dropped after the merge (its `KINDS` entry, methods, tests, and definition paragraph); the comparison report keeps its RCF discussion as a record of the choice.
