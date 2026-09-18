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
4. [x] **Simulate** in `AI_REMOTE_DIR=~/repros/ai-runs/switch-kexo-port` on Raven; fetch `per_round.parquet`. Ran in 1 min 44 s, exit 0; fetched copy is byte-identical to the cluster's (sha256 `d94f9a9b37cdfede`).
5. [x] **Evaluate** all 22 rows locally with `PYTHONPATH=<worktree>/src`; comparison table via `scripts/data_analysis/switch_kexo_port_compare.py` (reuses `curpun_rebaseline.py`'s helpers) into `plots/data_analysis/evaluation/switch_kexo_port/`.
6. [x] **Judge** under §2 with RCE protected; log, PR.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-18 | baseline: frontier stack, numeric joint-exodus switch (parent's stage D) | SC 1.4271, RCD 1.3091, RCE 0.8942 | 13/22 | 1.0357 | - |
| 2026-09-19 | candidate: same stack, switch slot swapped to the k-one-hot head | SC 1.3286, RCD 1.6894, RCE 0.9519 | 14/22 | 1.0314 | **[FAIL]** |

### The run

Raven, `AI_REMOTE_DIR=~/repros/ai-runs/switch-kexo-port`, one simulation job, 1 min 44 s, exit 0. `per_round.parquet` is 100 episodes x 24 rounds x 8 agents = 19,200 rows, the same shape as the baseline sim, and the fetched local copy is byte-identical to the cluster's (sha256 `d94f9a9b37cdfede5e0b65c0c85203ff1efdc46e4d898016b19cae2d93bb471f`). Evaluated with `PYTHONPATH=<worktree>/src python -m aimanager evaluate <config>`: 22 metric rows, RCE present, scoring 500 repeats at seed 42. The noise-ceiling denominators in `evaluation/scores.csv` are identical to the baseline's to full precision, so before and after are scored against the same ceiling.

### Per row

Full table with deltas and bands in `plots/data_analysis/evaluation/switch_kexo_port/compare_table.md`; `before` = `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun`, `after` = the same stack with only `switch_model` changed. Scores are multiples of the human noise ceiling; delta negative is an improvement.

| metric | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.860 | 0.841 | -0.019 | <= 1 -> <= 1 |
| CB | 0.788 | 0.804 | 0.016 | <= 1 -> <= 1 |
| CC | 0.889 | 0.829 | -0.060 | <= 1 -> <= 1 |
| CD | 0.810 | 0.807 | -0.003 | <= 1 -> <= 1 |
| CE | 1.057 | 1.031 | -0.026 | 1-2 -> 1-2 |
| CF | 0.828 | 0.840 | 0.012 | <= 1 -> <= 1 |
| CG | 1.554 | 1.101 | -0.452 | 1-2 -> 1-2 |
| SA | 0.785 | 0.789 | 0.004 | <= 1 -> <= 1 |
| SB | 1.006 | 0.898 | -0.108 | 1-2 -> <= 1 |
| SC | 1.427 | 1.329 | -0.099 | 1-2 -> 1-2 |
| PA | 0.660 | 0.617 | -0.043 | <= 1 -> <= 1 |
| PB | 0.969 | 0.948 | -0.021 | <= 1 -> <= 1 |
| PC | 0.907 | 0.898 | -0.009 | <= 1 -> <= 1 |
| PD | 0.722 | 0.736 | 0.013 | <= 1 -> <= 1 |
| RCA | 1.633 | 1.507 | -0.126 | 1-2 -> 1-2 |
| RCB | 1.545 | 1.396 | -0.149 | 1-2 -> 1-2 |
| RCC | 1.530 | 1.613 | 0.083 | 1-2 -> 1-2 |
| RCD | 1.309 | 1.689 | 0.380 | 1-2 -> 1-2 |
| RCE | 0.894 | 0.952 | 0.058 | <= 1 -> <= 1 |
| RSA | 1.070 | 1.644 | 0.574 | 1-2 -> 1-2 |
| RPA | 0.693 | 0.648 | -0.045 | <= 1 -> <= 1 |
| RPB | 0.847 | 0.773 | -0.075 | <= 1 -> <= 1 |
| mean | 1.0357 | 1.0314 | -0.0043 | |
| rows <= 1 | 13 | 14 | +1 | |

### RCE per band (protected row)

OLS slope of the next-round contribution change on punishment received, per contribution band, over punished non-full contributors.

| stage | slope 0-4 | slope 5-9 | slope 10-14 | slope 15-19 | signs | vs human |
|---|---|---|---|---|---|---|
| human | +0.140 | +0.104 | -0.077 | -0.161 | ++-- | |
| before | +0.095 | +0.020 | -0.058 | -0.160 | ++-- | ++-- (4/4) |
| after | +0.076 | +0.033 | -0.020 | -0.092 | ++-- | ++-- (4/4) |

Magnitude ratios after/before: 0.80, 1.65, **0.34**, 0.57.

### Gate outcome

- **Gate 1 (a band improvement on SC or RCD): FAILED.** SC improves by 0.099 but stays in 1-2 (it would need <= 1); RCD moves the wrong way, 1.309 -> 1.689, and also stays in 1-2. Neither target row changes band.
- **Gate 2 (22-row mean <= 1.1393): passed.** After mean 1.0314 against the baseline 1.0357, a change of -0.0043; rows <= 1 goes 13 -> 14.
- **RCE protected row: VIOLATED.** No band drop (0.894 -> 0.952, both <= 1) and all four human signs kept (++--, 4/4), but the 10-14 band slope falls from -0.058 to -0.020, i.e. to 34% of its baseline magnitude, below the half threshold. The 15-19 band at 57% survives the rule but is the second-largest erosion. Under §2 this is a failure whatever the gates say.

**Verdict: [FAIL].** Gate 1 missed on both target rows and the protected row violated; gate 2 alone is not sufficient.

### Attribution: head or pairing?

The question this branch was opened to answer. The same k-one-hot head appears in two stacks; the middle column is this run. Columns 1 and 2 hold the head fixed and change the contributor; columns 2 and 3 hold the contributor fixed and change the head.

| row | k-onehot + Gaussian-MLP contributor (#174, `d_kexo` post-fix) | k-onehot + stimulus-skip contributor (this run) | numeric switch + stimulus-skip contributor (baseline) |
|---|---|---|---|
| SC (larger group size) | 1.076 | 1.329 | 1.427 |
| RCD (switching pull) | 0.760 | 1.689 | 1.309 |
| RCE (response slope) | 0.705 | 0.952 | 0.894 |
| RSA (who leaves when punished) | 1.196 | 1.644 | 1.070 |
| SB (overall switch rate) | 0.931 | 0.898 | 1.006 |
| CG (contribution spread) | 1.665 | 1.101 | 1.554 |

**Measured.** Carrying the head into the frontier contributor reproduces none of the three headline switching numbers. SC moves 28% of the way from the baseline 1.427 toward the Gaussian pairing's 1.076 and stops at 1.329. RCD and RCE move in the *opposite* direction from the Gaussian pairing's values: RCD to 1.689 against 0.760, RCE to 0.952 against 0.705. RSA, which the declaration did not name, degrades most in relative terms (1.070 -> 1.644).

**Inference, and its limits.** The good switching numbers of the Gaussian stack are not a portable property of the one-hot head: the head alone does not carry them into a different contributor. But "they belong to the Gaussian contributor" overstates what one stack shows, and the row set splits in a way the declaration's separability premise did not anticipate.

Only SC and SB are close to pure switch-slot rows — they count group sizes and switch events, which the switch head decides. Both improved, SB by a band. CG, a contribution-spread row, improved sharply (-0.452); it is contributor-produced but strongly mediated by group composition, which is exactly what the head changes. So the head does carry real group-size structure, and it transfers.

RCD and RCE do not measure the switch head directly. RCD is the OLS slope of a switcher's *contribution change* on the gap to the receiving group: the head chooses who switches, the contributor chooses the dc. RCE is a pure contributor response row (dc on punishment received, per contribution band) that the head touches only by changing who ends up in which group and therefore who gets punished. Both are joint rows by construction, and they are precisely the two that failed. The premise in §1 that "the switch slot is largely separable from the contributor" holds for the group-composition rows and does not hold for the response rows — which is the more useful result of this branch than the gate outcome.

So the honest reading is the third option, not either of the two the declaration posed: the head carries the group-size behaviour it was designed for, and the Gaussian stack's low RCD / RCE were properties of that contributor's response dynamics under the group composition the head produced, not of the head. A single stack cannot separate "the head needs the Gaussian contributor's smoother contribution paths" from "the head interacts badly with the stimulus-skip contributor specifically"; both are consistent with this run and they imply different successors.

**No error bars.** This is one seed, one run, no repeats, so the small deltas carry no measured run-to-run spread. The gate-1 miss is safe (RCD moves +0.380 and neither target row comes near a band edge), but the RCE protected-row violation rests on a single run's band slope: -0.058 -> -0.020 over 1,313 observations in the 10-14 band, with no estimate of how much that slope moves between seeds. The violation is recorded as the rule reads, and a successor that wants to overturn it should re-run the band fit across seeds rather than argue about it.

## 4. Notes

1. **What needed porting.** The numeric joint-exodus head (experiment #171) is in this branch's ancestry; the one-hot variant is not. `git diff origin/auto/punisher-current-contribution origin/auto/switch-exodus-k-onehot -- src/aimanager/generic/joint_exodus.py` is exactly the `size_encoding` feature plus docstring re-pointing; the graph.py side is the `joint_exodus_size_encoding` kwarg, two assertions, the head constructor argument, the key-vs-head consistency check and the `to_save` entry. Without the port the artifact loads (`GraphNetwork.__init__` swallows the unknown key through `**_`, and the pickled head carries its 39-wide MLP) but its forward would feed a 23-wide input into a 39-wide layer. The train.py diff between the lineages is comment-only.
2. **Reuse, not retrain.** The switch model is fitted on the human data alone (`experiments/2group_8agent_50ep.csv`, flip-doubled; byte-identical across the lineages) with `joint_exodus_k_onehot.yml`, whose every hyperparameter equals this lineage's `joint_exodus.yml` (375 epochs, batch 10, lr 5e-4, weight decay 1e-3, hidden 10, seed 38381, `switch_every` 4) except the one-hot key and the output dir. Its inputs are game observables (common good, punishment received, group label, round), which the contributor swap does not change. Nothing in its training depends on which contributor it is later paired with, so the artifact is the same object whichever lineage's checkout trained it. What differs is the code that trained it: the `auto/switch-exodus-k-onehot` checkout (Gaussian-MLP lineage), whose graph.py lacks the group-vnode / stimulus-skip / herding-copula machinery of this lineage; for a switch model all of that is off, and the trunk (mlp+rnn+edge) is shared code. This is stated so the log is honest: the artifact was not produced on this branch, and the sha256 prefix recorded in the results identifies exactly which file was simulated.
3. **Artifact provenance check (measured).** `artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`: 17,284 bytes, zip archive, sha256 `28dd4b40fb5f72ab...`; loaded through the ported code on the Raven login node: saved key `onehot`, head `size_encoding` `onehot`, readout in_features 39, `joint_exodus_switch_every` 4. Baseline `switch_joint_exodus` artifact: 16,644 bytes, sha256 `8a4ae4ade60d5443...`, key `None`, head numeric, in_features 23.
4. **Reuse confirmed at the blob level (measured).** The port commit `81f3081` touches no file under `artifacts/`. The artifact's git blob is `c5fd8851a260c530c6689a67caf6c757279ec3f4`, and `git rev-parse origin/auto/switch-exodus-k-onehot:<path>` returns the same blob, so the file simulated here is byte-identical to the one experiment #174 trained. It first entered any branch in `3644698` ("Add the retrained one-hot switch artifact (step 6)") on `auto/switch-exodus-k-onehot`, and reached this lineage's ancestry in `fc4a643`, the punisher re-baseline commit that imported the gmlp-lineage artifacts. **The training context is therefore the `auto/switch-exodus-k-onehot` checkout (Gaussian-MLP lineage), not this branch**; nothing was retrained for this experiment, and `joint_exodus_k_onehot.yml` is carried here for provenance only. This is the honest provenance either way: the artifact is the same object, and the code that produced it is not this branch's.
5. **The evaluation was re-run, not inherited.** The predecessor's `evaluation/` directory contained only `metrics.csv`; `run_cli` writes metrics, then the 500-repeat scores, then the figures, so it had been cut off before scoring. The full suite was re-run here and produced `scores.csv` and 25 figures. The `PYTHONPATH=<worktree>/src` prefix is load-bearing and was used: the main checkout's `evaluation_suite/metrics.py` contains no `RCE` at all, so the shared venv would have scored 21 rows and silently dropped the protected row.
6. **Lint.** `graph.py`, `joint_exodus.py` and `switch_kexo_port_compare.py` are black- and flake8-clean. `black --check src/` also flags `rl_manager.py` and `artificial_humans/train.py`; both are pre-existing drift on the base branch (this branch changes zero lines in them) and were deliberately left alone to keep the diff to the experiment.

## 5. What this leaves for a successor

1. **The switch slot is not one slot.** The clean result here is the row split: group-composition rows (SC, SB, CG) follow the switch head across contributors, response rows (RCD, RCE, RSA) do not, because their dependent variable is contributor-produced. Any future switch-slot experiment should declare targets only from the first set, and treat RCD in particular as a joint row rather than a switch target — this branch declared it as a target and that was a mis-specification, independent of the outcome.
2. **The one-hot head is worth keeping for the composition rows, on its own merits.** In this pairing it buys SB a band (1.006 -> 0.898), CG -0.452, SC -0.099, RCA -0.126 and RCB -0.149 while the mean goes slightly down and rows <= 1 up by one. It fails here only because the declared targets and the protected row were the wrong rows for what it does. A re-run declaring SB / CG / SC as targets, with RCE still protected, would be a fair test of the same artifact and is cheap (one 2-minute sim).
3. **RSA is the unexplained degradation** (1.070 -> 1.644, the largest single regression). RSA is the switch share per received-punishment bin over punished contributors — closer to a pure switch-head row than RCD is, and it moved sharply the wrong way while SB and SC improved. That combination (right number of switches, right group sizes, wrong *who* leaves after punishment) is a concrete, diagnosable mismatch and the most informative single follow-up in this set.
4. **Distinguishing the two live explanations needs a third contributor.** "The head needs the Gaussian contributor's smoother paths" and "the head interacts badly with the stimulus-skip contributor" both fit this run. Pairing the same head with the PR #179 group-vnode contributor, or with the inflated gmlp of #177, separates them at the cost of one sim each.
5. **The 10-14 RCE band is fragile in both lineages.** It is the band the held-out teacher-forced test (PR #183) found was never learned, it is the band the punisher re-baseline moved most, and it is the band that broke the protection rule here. Whether the protection threshold should be a fixed half-magnitude on a band this weak (|slope| 0.058 at baseline, against a human -0.077) is a protocol question worth raising before it fails another experiment for a 0.038 move.
6. **Program step 3 is answered negatively; step 4 should not assume the merge.** The partial merge of the two lineages does not transfer the Gaussian stack's switching advantage. The remaining lineage-merge question (the emission head a sibling branch is testing) is unaffected by this result, but the premise that the two lineages' strengths are separable and additive now has one clear counter-example.
