# Autoresearch log: contribution-peer-attention

## Declaration

- **Slot:** contribution
- **Base model:** GNN (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`, the reference contributor)
- **Target rows** (reference severity-copula cell, concordant >= 2 in 10/10
  gnn-contribution contexts of `23_stack_sweep_severity_copula`):
  - **CG 9.808514** (band > 5; upgrade requires < 5)
  - **RCD 2.941928** (band 2-5; upgrade requires < 2)
  - **RCA 2.082907** (band 2-5; upgrade requires < 2)
- **Hypothesis:** Humans condition selectively on specific peers — their own
  group's level and its extremes — not on a uniform room average. The current
  `NodeModel` aggregates the 7 incoming peer messages with `scatter_mean`
  (uniform 1/7, both groups blended), so any single peer's signal is diluted
  and the model cannot concentrate on the peers a human actually reacts to.
  Learned attention over incoming messages lets conditioning concentrate where
  humans' does: on the receiving group after a switch (RCD, switching pull),
  on the own group's level for norm tracking (CG, free-running group
  coherence — PR #149 showed peer conditioning absorbs group dependence), and
  on the round context that drives contribution change (RCA).
- **Planned change:** single-head edge attention in `GraphNetwork.op1` — a
  scalar score per edge from the inputs the edge MLP already sees
  (`[x_src, x_dest, edge_attr, u]`), `scatter_softmax` over each node's
  incoming edges, weighted sum replacing `scatter_mean`. Flag-gated
  (`use_attention`, default off) so legacy artifacts stay bit-identical.
  The `same_group` edge feature (merged in PR #113) is supplied as edge input
  so attention can learn group-selective weighting. Variant selection
  (attention with/without same_group edge input) by Stage-1 target scores,
  per §5.

## Plan

Validated by the orchestrator (targets per §2, all steps legal per §5,
nothing on the frozen surface §8 — the `evaluation_sweep.py` marker edit in
step 16 is analysis-layer, not frozen).

- [x] 1. Worktree commit identity set (Claude / noreply@anthropic.com).
- [x] 2. Optional keyword-only `edge_weight=None` in `NodeModel.forward`
      (`src/aimanager/generic/graph.py`): `None` keeps the exact
      `scatter_mean` path; a weight switches to
      `scatter_add(edge_attr * edge_weight, col, ...)`. Legacy pickled
      `NodeModel`s hit the default and stay bit-identical.
- [x] 3. `EdgeAttention` module: `Lin(2*x + edge + u features -> 1)` scoring
      `[src, dest, edge_attr, u[batch]]`, `scatter_softmax` over each
      destination's incoming edges, returns `(E, n_rounds, 1)` weights.
- [x] 4. `AttentionMetaLayer`: mirrors `MetaLayer.forward`; scores from the
      pre-update `x`/`edge_attr`, passes `edge_weight=alpha` to the node
      model. Separate class so legacy artifacts keep restoring `MetaLayer`.
- [x] 5. Flag-gate `use_attention=False` in `GraphNetwork.__init__`; add to
      `save()`'s `to_save` list; `load()` unchanged (legacy files omit the
      key and default off).
- [x] 6. Raven unit tests `src/aimanager/tests/test_graph_attention.py`:
      off-flag equals scatter_mean exactly; alpha sums to 1 per destination;
      zeroed score head reproduces the mean; alpha responds to same_group;
      save/load round-trip + legacy-checkpoint load.
- [x] 7. Run tests on Raven (`scripts/remote_test.sh`), with the
      pending-job `squeue` check before the sync.
- [x] 8. Two training configs, exact copies of
      `configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml`
      plus `use_attention: true`: `peer_attention_sg.yml` (adds
      `edge_encoding: same_group`) and `peer_attention_nosg.yml` (no edge
      encoding); slugged output dirs `contribution_peer_attention_{sg,nosg}`.
- [x] 9. Train both on Raven (`scripts/train_cluster.sh ah ...`), pending-job
      check first. Wall clock ~1-4 h each, parallel jobs.
- [ ] 10. Fetch artifacts; log per-fold test log_loss vs the reference's
      1.9897 (sanity, not a gate); commit artifacts.
- [ ] 11. Two Stage-1 sim configs copying the reference sim config with only
      the contribution artifact path + slugged output dirs swapped; naming
      keeps the `..._self_<contr>_contr_<switch>_switch` pattern.
- [ ] 12. Simulate both on Raven (~3 min each).
- [ ] 13. Fetch, `python -m aimanager evaluate` locally, append Results rows
      (unrounded) vs the known baseline (no baseline re-run).
- [ ] 14. Decision gate: better variant by Stage-1 targets (tie -> simpler,
      no same_group); kept iff targets improve, rows<=1 >= 10, mean <=
      1.687998; Stage 2 only on a band upgrade (CG < 5, RCD < 2, or RCA < 2).
- [ ] 15. Interpretability analysis (not a gate):
      `scripts/data_analysis/peer_attention_weights.py` — alpha on same-group
      vs other-group edges, alpha vs peer extremeness, entropy vs uniform 1/7.
- [ ] 16. Stage 2 if band upgrade: candidate in the contribution slot across
      the sweep family, extend `CONTR_ORDER`/`CONTR_MARKERS` in
      `scripts/data_analysis/evaluation_sweep.py`, run the sweep, check the
      slot claim across contexts.
- [ ] 17. Close out: complete the log, open the `[SUCCESS]`/`[FAIL]` PR
      (Hypothesis / Results / Collateral).

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## Notes

1. Direction chosen by the maintainer from four candidates (persistent-type
   mixture, behavioral-mode head, peer attention, scheduled sampling);
   scheduled sampling is already claimed by the parallel
   `auto/contribution-cg-schedsamp` branch.
2. Back-compat verified two ways before training: the full Raven suite (97
   tests incl. 7 new) passes, and the zeroed-score-head test pins the
   attention path to reproduce the uniform mean exactly. `save()` pickles
   module objects, so the `AttentionMetaLayer` is a separate class and
   legacy checkpoints restore the unmodified `MetaLayer` untouched.
3. Training jobs submitted 2026-08-13: 29319275 (sg), 29319286 (nosg),
   identical recipes to the reference (seed 38381, 575 epochs); only
   `use_attention` and (sg) the `same_group` edge feature differ.
