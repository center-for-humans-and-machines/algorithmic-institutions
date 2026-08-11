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

(to be filled by the validated step list)

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
