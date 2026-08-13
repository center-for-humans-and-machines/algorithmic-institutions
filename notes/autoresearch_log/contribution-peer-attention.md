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

(to be filled by the validated step list)

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## Notes

1. Direction chosen by the maintainer from four candidates (persistent-type
   mixture, behavioral-mode head, peer attention, scheduled sampling);
   scheduled sampling is already claimed by the parallel
   `auto/contribution-cg-schedsamp` branch.
