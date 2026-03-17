# [DRAFT] Repository cleanup

Tracks issue [#52](https://github.com/center-for-humans-and-machines/algorithmic-institutions/issues/52).

## Goal

Remove stale branches, unused artifacts, legacy experiment data, and
other accumulated bloat to improve repo navigability and reduce size.

---

## Phase 1 — Git housekeeping (branches, issues, PRs)

### 1a. Close stale issues & PRs

- [x] Close issue #23 (pseudo group matching — done, merged via PR #51)

### 1b. Delete remote branches merged into main

- [x] `12-investigating-behavioral-changes-introduced-by-pr-6`
- [x] `21-group-id-as-additional-node-feature-for-artificial-humans`
      (already deleted remotely, pruned tracking ref)
- [x] `autoregression`
- [x] `batch_env`
- [x] `claude-code-setup`
- [x] `composit_q_value`
- [x] `graph_neural`
- [x] `multi_group_simulation`
- [x] `multi_manager_rl`
- [x] `optimize_payoff`
- [x] `rewards_per_group`
- [x] `tidy`

### 1c. Delete remote branches merged into claude-code-track

- [x] `47-train-skill`
- [x] `49-simulate-skill`
- [x] `53-unified-cli`

### 1d. Delete remote branches from closed/superseded PRs

- [x] `claude-code-init` (already deleted, pruned tracking ref)
- [x] `35-integrate-project-to-claude-code` (already deleted, pruned
      tracking ref)

### 1e. Delete remote branches for completed work

- [x] `23-pseudo-group-matching` (PR #51 merged)
- [x] `23-pseudo-group-matching-clean` (PR #51 merged)

### 1f. Kept — branches with active issues

Remaining remote branches have active issues linked and are kept:
`14-parallel-group-training`, `15-relax-stable-group-assumption`,
`41-reward-timing`, `43-group-switching-predictor`, `experiment`,
`power_analysis`, `feature/temperature`, `mae_accuracy`,
`transformer`, `slim`, `aggregate_reward`,
`enable_dynamic_groups`, `redesign_env`.

### 1g. Delete stale local branches

- [x] `23-pseudo-group-matching`
- [x] `23-pseudo-group-matching-clean`
- [x] `sync-from-cluster`
- [x] Switched to `claude-code-track` and deleted local `53-unified-cli`

## Phase 2 — Skipped (artifacts kept as bookkeeping/history)

## Phase 3 — Skipped (experiment data kept as bookkeeping/history)

---

## Phase 4 — Unused configs, plots, and reports

### 4a. Remove old training configs

Keep: `pseudo_group.yml`, `pseudo_group_combined.yml`,
`script_21_no_grid.yml`, `rl_manager/01_rnn_node.yml`.
Keep all simulation configs.

- [x] `configs/training/artificial_humans/notebook_21.yml`
- [x] `configs/training/artificial_humans/notebook_22.yml`
- [x] `configs/training/artificial_humans/group_switching_21.yml`
- [x] `configs/training/artificial_humans/script_21.yml`
- [x] `configs/training/artificial_humans/script_22.yml`
- [x] `configs/training/artificial_humans/test_small.yml`

### 4b. Remove old plot directories

- [x] `plots/simulation/ah_training_eval/`
- [x] `plots/simulation/pr5_4/`
- [x] `plots/simulation/pr6/`

Keep: `plots/simulation/ah_group_testing_8/`,
`plots/simulation/ah_group_testing_8_combined/`,
`plots/simulation/pr7_n_groups==1/`,
`plots/simulation/pr7_n_groups==2/`,
`plots/behavioral_cloning/`, `plots/exp2/`,
`plots/group_selection/`, `plots/key_metrics.png`.

### 4c. Remove old reports (verify first)

- [ ] `reports/short_verison.md` (duplicate content)
- [ ] `reports/rule_base_manager.md` (abandoned approach)
- [ ] `reports/exp1_manager_training_and_eval.md` (old experiment)
- [ ] `reports/draft_publishable.md` (outdated draft)
- [ ] `reports/comparision.md` (old comparison notes)

Keep: `basics.md`, `artificial_humans.md`, `manager.md`,
`humanlike_manager.md`, `up_to_date_docs.md`.

---

## Phase 5 — Miscellaneous

- [x] Delete `update_artifacts.py` (superseded by CLI + fetch skill)
- Kept `notebooks/archive/` for reference
- Skipped `doc/plans/47-train-skill.md` — still ACTIVE

---

## Phase 6 — Config consistency

- [ ] Verify all remaining configs reference only kept artifacts
- [ ] Update simulation configs if artifact paths changed in phase 2e
- [ ] Verify `.artifactinclude` matches final artifact set
- [ ] Sync cleaned artifacts to Raven cluster

---

## Notes

- LFS objects remain in git history even after removing from HEAD.
  To reclaim space, `git filter-repo` or BFG would be needed — out of
  scope for this pass.
- Reports in 4c should be verified with the team before deletion.
- Phase 1f branches need human decision on each.
