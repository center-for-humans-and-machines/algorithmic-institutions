# Experiment log: contribution-cg-selfdrop

## Declaration

- **Slot:** contribution
- **Base model:** GNN (node+edge+rnn, `group_switching_contribution_50ep`
  lineage; reference-stack artifact = the M0 model)
- **Target row:** CG (reference stack baseline **9.850**, the stack's worst
  row; deficit profile across contr=gnn contexts: CG 9.65 averaged over the
  other two slots, concordant across all 8 contexts)
- **Provenance disclosure:** this direction was discovered **post-hoc** in the
  runs of the failed experiment `contribution-self-history-dropout` (PR #144,
  declared at RCA+RCD). The Stage-1 numbers quoted below are that
  experiment's runs — **prior evidence, not confirmation**. This experiment
  declares CG up front and lets Stage 1 (the new arm) and Stage 2 decide.
- **Hypothesis:** Humans in a group converge on a shared contribution norm
  (conditional cooperation), so the spread of group means sits well above the
  independence floor (human CG ratio 0.848 vs floor ~0.55). The M0 GNN
  free-runs at the floor (sim ratio 0.579) because it anchors on its own
  previous contribution and barely uses group context (free-running peer
  β ≈ 0.03 vs human 0.205; the cat linear reaches 0.172 and CG ≈ 1.9–2.4,
  proving observable conditioning alone can carry near-human group
  coherence). Giving the model the peer signal (#114 M4 features:
  `own_grp_prev_mean_contr` node + `same_group` edge) and weakening the
  self-anchor with training-time input dropout on `prev_contribution` shifts
  conditional load onto shared group context; agents in a group then act
  more alike and group means diverge — the between-participant correlation
  CG measures.
- **Behavioral rationale (one sentence):** Humans contribute like their
  current group does, not only like their past selves, so loading the
  conditional distribution onto shared group context should widen the spread
  of group means toward the human ratio (CG).
- **Planned change (one change):** the M4 feature set with training-time
  input dropout on `prev_contribution` at rate p ∈ {0, 0.10, 0.15}, selected
  by Stage-1 CG (legal variant selection, §5). The p=0.10 arm is newly
  trained; p=0 (the #114 M4 artifact) and p=0.15 (PR #144's
  `auto_contribution_selfdrop_15` artifact) reuse the existing artifacts and
  their Stage-1 runs as prior evidence — same configs, same seeds, no
  re-running for a better draw (§10).
- **Prior evidence (PR #144 Stage-1 runs, reference stack):**
  | arm | CG | rows <= 1 | mean | known cost |
  |---|---|---|---|---|
  | baseline (M0) | 9.850 | 11/21 | 1.760 | — |
  | M4, p=0 | 7.587 | 11/21 | 1.621 | none observed (RCB +0.21, RCD within noise) |
  | M4, p=0.15 | 5.961 | 12/21 | 1.585 | RCA 2.035 → 2.772, RCB 1.928 → ~2.65 |
  The RCA/RCB cost is entirely from the dropout (p=0 sits at reference
  level); CG splits roughly half features / half dropout. p=0.10 probes
  whether most of the dropout's CG gain arrives before most of its RCA/RCB
  cost (the cost is monotone in p; the gain plateaus by p=0.15).
- **Verdict criteria:** Stage 1 in the reference stack (gnn switch x
  lin_multinomial punisher, 23-family protocol: 2x8, 24 rounds, 100
  episodes, seed 42). Keep iff CG improves vs 9.850, rows<=1 does not fall
  below 11/21, mean does not rise above 1.760. Winner by Stage-1 CG subject
  to those guards, ties to the simpler model (§5: p=0 beats p>0 on ties).
  Then Stage 2: the 8-config family with the candidate replacing the gnn
  contribution slot everywhere, `evaluation_sweep.py`, and the Kendall's-W
  slot discipline of PR #143 — the claim stands only if the candidate wins
  its slot across contexts.

## Plan

Validated 2026-08-11. Constraints discovered during planning, binding on all
steps: (A) `evaluation_sweep.py` hardcodes `CONTR_ORDER` and silently
*averages* two dirs parsing to the same (contr, switch) key, so candidate sim
dirs are named `..._self_gnn_contr_{gnn,lin}_switch` — in this experiment's
Stage-2 matrix the label `gnn` means the candidate, not M0 — and the baseline
`23_2g8a_self_gnn_contr_*` dirs must never be passed to the sweep alongside
them. (B) Kendall's W is only rendered into `slot_concordance.jpg`; the log
quotes it by recomputing from `score_matrix.csv` with the script's formula.
(C) the `self_dropout` label must be a quoted string (`'0.10'`) — it is
interpolated into the `.pt` filename. (D) the p=0 (M4) `.pt` exists only on
Raven; fetch before committing if it wins. (E) all cluster scripts rsync
`--delete` into the shared `~/algorithmic-institutions`; check no parallel
experiment has jobs in flight before any sync, and re-sync from this worktree
immediately before each submission. (F) `train_cluster.sh`/
`simulate_cluster.sh` return at submission; every cluster step polls
`squeue` + output files before being confirmed. (G) if p=0 or p=0.15 wins,
both of its Stage-2 cells are simulated fresh under slug names (identical
configs, seed 42 — not seed shopping); the Stage-1 prior-evidence numbers
stay as declared, and any reproduction mismatch is logged.

- [x] 1. Record this plan in the log; commit.
- [x] 2. Verify the cherry-picked dropout mechanism on Raven
      (`scripts/remote_test.sh -- -k input_dropout -v`; PyG import, cannot
      run locally). Confirm: 5 passed.
- [x] 3. Add `configs/training/artificial_humans/contribution/
      auto_cg_selfdrop_10.yml` — copy of the predecessor's
      `auto_selfdrop_15.yml` with exactly: description, label
      `self_dropout: '0.10'`, `input_dropout.prev_contribution: 0.10`,
      `output_dir: artifacts/artificial_humans/auto_contribution_cg_selfdrop_10`.
      Commit.
- [ ] 4. Train on Raven (`scripts/train_cluster.sh ah <config>`); poll.
      Confirm: `.pt` with the exact expected name exists on Raven.
- [ ] 5. Fetch the artifact; sanity-check fold-mean best test log-loss lands
      between p=0 (1.9899) and p=0.15 (2.0128). Notes entry; commit.
- [ ] 6. Add Stage-1 sim config `configs/simulation/manager_testing/
      auto_cgselfdrop10_2g8a_self_gnn_contr_gnn_switch.yml` — 23-family
      template, only the contribution path + output_dir/figure_name changed;
      doubles as the Stage-2 gnn-switch cell. Commit.
- [ ] 7. Simulate on Raven; fetch. Confirm: `per_round.parquet` +
      `aggregates.csv` with all four pairings.
- [ ] 8. Evaluate locally; read CG / rows<=1 / mean from the
      `lin_multinomial_self` run (the reference stack). Results row; commit.
- [ ] 9. Select the winner per the declaration (lowest Stage-1 CG, guards
      rows<=1 >= 11/21 and mean <= 1.760, ties to simpler). No passing arm →
      [FAIL], skip to 14. If p=0.15 wins, add its training config verbatim as
      a provenance copy. Notes entry; commit.
- [ ] 10. Add the winner's remaining Stage-2 sim config(s): always the
      lin-switch cell; also the gnn-switch cell if the winner is p=0/p=0.15.
      Commit.
- [ ] 11. Simulate on Raven, fetch, evaluate each new cell. Confirm: every
      candidate dir has `evaluation/scores.csv` (84 rows).
- [ ] 12. Stage-2 sweep: `evaluation_sweep.py
      auto_contribution_cg_selfdrop_stage2` over exactly 8 dirs — the 2
      candidate cells + the 6 existing `23_*` cat/gaussian/ridge dirs, never
      the baseline gnn dirs. Confirm: 32-row `score_matrix.csv`.
- [ ] 13. Analyse Stage 2 vs `23_stack_sweep_updated/score_matrix.csv`:
      per-context CG deltas, RCA/RCB cost, recomputed Kendall's W and
      `best in n/8` for the contribution slot. Stage-2 Results row + Notes;
      commit results, sim dirs, sweep outputs, and the winner's `.pt`.
- [ ] 14. Open the PR (`[SUCCESS]`/`[FAIL]`), body per §9: Hypothesis /
      Results / Collateral. Final Notes entry records the verdict.

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-11 | (baseline) reference stack, no change | 1 | CG 9.850 | 11/21 | 1.760 | baseline |
| 2026-08-11 | M4 features, p=0 (PR #144 run, prior evidence) | 1 | CG 7.587 | 11/21 | 1.621 | prior evidence |
| 2026-08-11 | M4 + selfdrop p=0.15 (PR #144 run, prior evidence) | 1 | CG 5.961 | 12/21 | 1.585 | prior evidence |

## Notes

1. 2026-08-11: Declared as the follow-up recommended in
   `contribution-self-history-dropout` note 8. The dropout mechanism commit
   (`train.py` + tests) is cherry-picked from that branch; the mechanism is
   config-gated and off by default, so the cherry-pick alone changes no
   behavior.
