# Experiment log: contribution-self-history-dropout

## Declaration

- **Slot:** contribution
- **Base model:** GNN (node+edge+rnn, `group_switching_contribution_50ep`
  lineage)
- **Target rows:** RCA, RCD (declared collectively; CG is the hoped-for
  side-effect, tracked but not a declared target)
- **Deficit profile** (from `23_stack_sweep_updated/score_matrix.csv`,
  contr=gnn contexts averaged over the other two slots): CG 9.65, RCA 2.85,
  RCD 2.62, RCB 2.08. In the reference stack: CG 9.85, RCD 2.77, RCA 2.03.
- **Hypothesis:** The GNN fails the reaction rows because behavioral cloning
  lets it anchor on the agent's own previous contribution and ignore its
  current group's level: PR #116 measured own-prior beta +0.74..0.78 vs human
  +0.45, and new-group-peer beta +0.014..0.031 vs human +0.247 -- *even when
  the leave-one-out own-group mean was handed to it as a feature* (the #114
  M3/M4 arms). The peer signal is present but unused because self-history is
  the cheaper predictor. Stochastically replacing `prev_contribution` with its
  round-0 default during training removes the shortcut on a fraction of
  agent-rounds and forces the conditional distribution to load on group
  context -- the conditional-cooperation mechanism RCA (round-type reactions)
  and RCD (switching pull) test. If group context drives the conditional mean
  harder, group means also diverge more, which is the direction CG needs
  (currently at the independence floor).
- **Behavioral rationale (one sentence):** Humans are conditional cooperators
  who adapt their contribution toward their current group's level, so forcing
  the model off its self-anchor and onto the peer signal should move RCA and
  RCD.
- **Planned change (one change):** training-time input dropout on
  `prev_contribution` (per agent-round Bernoulli, masked value = the feature's
  round-0 default, training batches only -- simulation unchanged), trained on
  the existing #114 M4 feature set (`own_grp_prev_mean_contr` node feature +
  `same_group` edge feature, both merged on main). Dropout rate p in
  {0.15, 0.3, 0.5}, selected by Stage-1 score (legal variant selection, §5).
- **Verdict criteria:** Stage-1 in the reference stack (gnn switch x
  lin_multinomial punisher, 23-family protocol). Keep iff RCA+RCD improve vs
  RCA 2.03 / RCD 2.77, rows<=1 does not fall below 11/21, mean does not rise
  above 1.76. Then Stage 2 (full 8-config sweep).

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-11 | (baseline) reference stack, no change | 1 | RCA 2.035, RCD 2.772 | 11/21 | 1.760 | baseline |
| 2026-08-11 | M4 features + prev_contribution dropout p=0.15 | 1 | RCA 2.772, RCD 2.336 | 12/21 | 1.585 | failed (RCA regressed) |
| 2026-08-11 | M4 features + prev_contribution dropout p=0.30 | 1 | RCA 3.223, RCD 2.948 | 11/21 | 1.696 | failed (both targets regressed) |
| 2026-08-11 | M4 features + prev_contribution dropout p=0.50 | 1 | RCA 4.142, RCD 2.791 | 11/21 | 1.690 | failed (RCA regressed) |
| 2026-08-11 | control: M4 features, no dropout (p=0) | 1 | RCA 2.126, RCD 3.042 | 11/21 | 1.621 | control (not a candidate) |

## Notes

1. 2026-08-11: Reviewed prior art before declaring. #112 (same_group edge) and
   #114 (own-group LOO mean) added the peer signal; PR #116 showed it barely
   moves simulated behavior (peer beta +0.031 vs human +0.247) -- so a plain
   artifact swap of the M3/M4 arms into the reference stack is unlikely to
   clear the keep bar, and the experiment attacks the *use* of the signal, not
   its availability. The M4 feature set is reused as the substrate because
   dropout on the self-anchor only helps if an intact peer signal exists to
   fall back on (the LOO mean excludes self, so it survives the masking).
2. 2026-08-11: Implementation choices. Masked cells are set to the feature's
   round-0 default (c_def via the `prev_`-to-base lookup), so "dropped" looks
   exactly like the "no history yet" state the model already handles -- no new
   out-of-range value, no train/sim distribution mismatch beyond the intended
   one. Dropout applies to the shared `prev_contribution` tensor, so peers'
   edge-view of a masked agent degrades too; accepted, because the intact
   `own_grp_prev_mean_contr` (precomputed in preprocessing) remains the
   reliable peer channel -- which is the channel we want the model to learn.
   Epochs kept at the 575 budget for comparability; dropout regularizes, so
   the known late-overfit risk shrinks rather than grows. Rates {0.15, 0.30,
   0.50}, one config each (`configs/training/artificial_humans/contribution/
   auto_selfdrop_{15,30,50}.yml`), Stage-1 sim configs
   `configs/simulation/manager_testing/auto_2g8a_self_selfdrop{15,30,50}_contr_gnn_switch.yml`
   (exact 23-family template, contribution path swapped).
3. 2026-08-11: Trained (Raven jobs 29266994-96, ~8 min each, seed 38381,
   5-fold + full). Fold-mean best test log-loss: M0 1.9892, M4 1.9899,
   selfdrop 0.15 -> 2.0128, 0.30 -> 2.0605, 0.50 -> 2.1538. Log-loss
   degrades monotonically with p, as expected -- the masking deliberately
   discards the most predictive input on a fraction of cells. This is the
   planned trade (likelihood for mechanism); whether it pays is decided by
   the Stage-1 scores, not by log-loss. Stage-1 sims submitted (jobs
   29267198/206/207).
4. 2026-08-11: Stage-1 verdict: **the declared hypothesis failed** -- RCA got
   *worse* in every variant, monotonically in p (2.035 -> 2.772 / 3.223 /
   4.142), and RCB moved with it (1.928 -> ~2.65). Diagnosis: RCA/RCB measure
   contribution *change*; weakening the self-anchor makes the model's own
   level noisier, so its change distributions degrade -- the dropout attacks
   the very statistic those rows condition on. RCD improved only at p=0.15
   (2.772 -> 2.336) and not monotonically, so the switching-pull gain is not
   clearly the mechanism either.
5. 2026-08-11: The same runs surfaced an undeclared but robust effect in the
   opposite family: the group-agreement rows improved across all three rates
   -- CG 9.850 -> 5.961/6.940/5.899 (the reference stack's worst row, cut by
   ~40%), SC 3.270 -> 2.767/2.720/2.402, CC 1.61 -> ~1.0, CE 1.33 -> ~1.0.
   Mechanically consistent: agents that lean less on their own history and
   more on shared group context act more alike, which is exactly the
   between-participant correlation the independence-floor rows measure (§6 of
   the guideline). At p=0.15 the stack metrics beat the baseline outright:
   rows<=1 12/21 (vs 11), mean 1.585 (vs 1.760). Per the declaration this is
   still a discard -- targets were RCA+RCD -- but it is the strongest CG
   movement any contribution change has produced, and it did not come from
   degrading the marginals much (CA 0.99, CD 0.87 at p=0.15, still at/near
   the ceiling).
6. 2026-08-11: Submitted a p=0 control (the #114 M4 model, same features, no
   dropout; sim job under auto_2g8a_self_selfdrop00_contr_gnn_switch) to
   attribute the CG/SC gains: if M4-without-dropout already shows them, the
   features (not the dropout) are the cause; if not, the dropout is doing the
   work. Result informs the follow-up declaration either way.
7. 2026-08-11: Control result -- the attribution splits cleanly in two:
   - **CG**: roughly half from the features (9.850 -> 7.587 at p=0), half
     from the dropout (7.587 -> 5.961 at p=0.15). **SC**: entirely from the
     features (3.270 -> 2.707 at p=0; flat in p until 0.50).
   - **RCA/RCB degradation**: entirely from the dropout (p=0 sits at the
     reference level: RCA 2.126, RCB 2.138). **RCD** is noise at Stage-1
     resolution (3.042 / 2.336 / 2.948 / 2.791 across p -- spread ~0.7 with
     no trend; treat RCD Stage-1 deltas < ~0.7 as unreadable).
   - The M4 feature set alone (p=0) is a quiet across-the-board win the #114
     evaluation missed because the eval suite did not exist yet: mean 1.760
     -> 1.621, CG -2.26, SC -0.56, CC/CE/CA/SA all better or equal; worst
     collateral RCB +0.21, RCD +0.27 (within RCD's noise). rows<=1 unchanged.
8. 2026-08-11: **Final verdict: FAILED as declared** (targets RCA+RCD did not
   improve; RCA regressed monotonically in p). Not reverting the code -- the
   input-dropout mechanism is sound, config-gated, and off by default -- but
   no candidate from this experiment goes to Stage 2 under this declaration.
   **Recommended follow-up** (next loop iteration, own branch + declaration):
   target **CG** with the M4 feature set, p in {0, 0.10, 0.15} -- at p=0.15
   the stack metrics already beat the baseline outright (rows<=1 12/21, mean
   1.585), and at p=0 nothing regresses at all. Disclose in that declaration
   that the direction was discovered post-hoc in this experiment's runs (the
   Stage-1 numbers here are its prior evidence, not its confirmation); the
   §6 warning applies -- the RCA/RCB cost of nonzero p is real, so the
   Kendall's-W discipline of Stage 2 should decide whether the trade holds
   across contexts.
