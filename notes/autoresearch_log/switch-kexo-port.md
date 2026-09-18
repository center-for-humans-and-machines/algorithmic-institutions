# Port the k-one-hot joint-exodus switch head into the frontier stack

## 1. Declaration

**Slot:** switch.

**Parent:** PR #184 (`auto/punisher-current-contribution`, head `01f966a`): the punisher fixed to the current round's contribution, the RCE response-slope row present and protected, 22-row suite. PR opens with `--base auto/punisher-current-contribution`.

**Program context:** step 3 of a four-step program merging the two model lineages. The graph-network lineage (PR #179 -> #181 -> #184) carries the stimulus-skip contributor and the numeric joint-exodus switch; the Gaussian-MLP lineage (`auto/switch-exodus-k-onehot`, experiment #174, and its child #177) carries the one-hot group-size joint-exodus switch. The two lineages diverged by about 17 files. Siblings in flight in parallel: `auto/head-state-spread-diagnostic`, `auto/punisher-ceiling-fix`.

**Base model (this lineage's switch):** `artifacts/artificial_humans/switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` (numeric joint-exodus head, `size_encoding` numeric, readout MLP 23 wide, `joint_exodus_switch_every` 4, `x_encoding` common_good / punishment / agent_group / round_number).

**Candidate:** `artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` (experiment #174's artifact: same trunk, same data, same hyperparameters and seed, joint head with `size_encoding` onehot, readout MLP 39 wide = 2x10 pooled + 2x9 one-hot + round). Trained on the `auto/switch-exodus-k-onehot` branch, i.e. the Gaussian-MLP lineage's checkout; reused here, not retrained (note 2).

**Evaluation stack (§3 under the parent rule of §9):** the parent's re-baselined frontier stack, `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun`: vnode GNN contributor with the immediate-stimulus skip and its calibrated copula, current-contribution lin_multinomial punisher with its severity copula, single pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds, `save_per_round: true`. Candidate config: `configs/simulation/manager_testing/23_2g8a_switch_kexo_port_self_gnncopar1_contr_stimulus_skip_contr_gnn_kexo_switch_curpun.yml`, differing from the source config in `switch_model`, `output_dir` and `figure_name` only. No copula is recalibrated: copula parameters are frozen per model family for this program.

**Baseline (the parent's post-fix scores; both gates judged against these; full precision in `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun/evaluation/scores.csv`):**

| quantity | value |
|---|---|
| SC | 1.4271261720677766 (band 1-2) |
| RCD | 1.309133193875904 (band 1-2) |
| RCE (protected) | 0.8942480256952102 (band <= 1); band slopes +0.095 / +0.020 / -0.058 / -0.160, all four human signs |
| mean over 22 rows | 1.0357 |
| rows <= 1 | 13/22 |

**Target rows:** SC (1.427, band 1-2 -> requires <= 1) and RCD (1.309, band 1-2 -> requires <= 1). **Gate 2:** 22-row mean <= 1.10 x 1.0357 = 1.1393. **Protected row RCE:** no band drop, no band slope losing the human sign (+ + - -), no band slope magnitude falling to half its baseline or below.

### Hypothesis

The stack built around the k-one-hot switch posts the best switching numbers in the whole re-baselined set (SC 1.076, RCD 0.760, RCE 0.705 in `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun`), but its contributor fails elsewhere (CA 1.703, RCA 3.428). The switch slot is largely separable from the contributor: the switch model conditions on the same game observables whoever generated them. If the switching numbers belong to the head, swapping only the switch model into the frontier stack should move SC and RCD down a band while the contributor rows stay where they are. If they belong to the Gaussian contributor's dynamics (its smoother contribution paths feeding the switch head), SC and RCD will not move, and the good numbers were a property of the pairing.

Behavioral rationale of the head itself (from experiment #174): humans never empty a group of five or more (0 of 119 cells) and empty singletons in 16.1% of cells, while a numeric `k / 8` scalar through one Tanh unit can only bend one smooth curve through that hump and floor; the one-hot code gives the head nine free intercepts per group label so the size dependence is free-form. That is the segregation row SC (group-size dynamics) and the switching-pull row RCD.

## 2. Plan

1. [x] **Port the code path** (`src/aimanager/generic/joint_exodus.py`, `src/aimanager/generic/graph.py`): `JointExodusHead.size_encoding` (numeric | onehot) with a `__setstate__` default so pre-existing heads keep their 23-wide MLP; `GraphNetwork(joint_exodus_size_encoding=...)`, its assertions, the consistency check between the saved key and the pickled head, and the key in `save()`. Taken from `origin/auto/switch-exodus-k-onehot`, doc references re-pointed to this lineage's logs. Training config `configs/training/artificial_humans/switch_predictor/joint_exodus_k_onehot.yml` brought over for provenance. Implementer: Fable (this agent).
2. [x] **Verify the artifact** is a real file (17,284 bytes, zip archive, not an LFS pointer) and loads through the ported code on Raven: key onehot, head onehot, MLP in 39, `switch_every` 4. The baseline artifact still loads as numeric / 23.
3. [x] **Candidate sim config**: frontier config with only `switch_model` swapped.
4. [ ] **Simulate** in `AI_REMOTE_DIR=~/repros/ai-runs/switch-kexo-port` on Raven; fetch `per_round.parquet`.
5. [ ] **Evaluate** all 22 rows locally with `PYTHONPATH=<worktree>/src`; comparison table via `scripts/data_analysis/switch_kexo_port_compare.py` (reuses `curpun_rebaseline.py`'s helpers) into `plots/data_analysis/evaluation/switch_kexo_port/`.
6. [ ] **Judge** under §2 with RCE protected; log, PR.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-18 | baseline: frontier stack, numeric joint-exodus switch (parent's stage D) | SC 1.4271, RCD 1.3091, RCE 0.8942 | 13/22 | 1.0357 | - |

## 4. Notes

1. **What needed porting.** The numeric joint-exodus head (experiment #171) is in this branch's ancestry; the one-hot variant is not. `git diff origin/auto/punisher-current-contribution origin/auto/switch-exodus-k-onehot -- src/aimanager/generic/joint_exodus.py` is exactly the `size_encoding` feature plus docstring re-pointing; the graph.py side is the `joint_exodus_size_encoding` kwarg, two assertions, the head constructor argument, the key-vs-head consistency check and the `to_save` entry. Without the port the artifact loads (`GraphNetwork.__init__` swallows the unknown key through `**_`, and the pickled head carries its 39-wide MLP) but its forward would feed a 23-wide input into a 39-wide layer. The train.py diff between the lineages is comment-only.
2. **Reuse, not retrain.** The switch model is fitted on the human data alone (`experiments/2group_8agent_50ep.csv`, flip-doubled; byte-identical across the lineages) with `joint_exodus_k_onehot.yml`, whose every hyperparameter equals this lineage's `joint_exodus.yml` (375 epochs, batch 10, lr 5e-4, weight decay 1e-3, hidden 10, seed 38381, `switch_every` 4) except the one-hot key and the output dir. Its inputs are game observables (common good, punishment received, group label, round), which the contributor swap does not change. Nothing in its training depends on which contributor it is later paired with, so the artifact is the same object whichever lineage's checkout trained it. What differs is the code that trained it: the `auto/switch-exodus-k-onehot` checkout (Gaussian-MLP lineage), whose graph.py lacks the group-vnode / stimulus-skip / herding-copula machinery of this lineage; for a switch model all of that is off, and the trunk (mlp+rnn+edge) is shared code. This is stated so the log is honest: the artifact was not produced on this branch, and the sha256 prefix recorded in the results identifies exactly which file was simulated.
3. **Artifact provenance check (measured).** `artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`: 17,284 bytes, zip archive, sha256 `28dd4b40fb5f72ab...`; loaded through the ported code on the Raven login node: saved key `onehot`, head `size_encoding` `onehot`, readout in_features 39, `joint_exodus_switch_every` 4. Baseline `switch_joint_exodus` artifact: 16,644 bytes, sha256 `8a4ae4ade60d5443...`, key `None`, head numeric, in_features 23.
