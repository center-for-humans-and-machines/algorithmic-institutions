# Logistic-regression baselines for the three AH models (50ep)

Each AH GNN is compared against (a) a **constant floor** (predict the train-set
marginal class distribution) and (b) a **logistic regression** trained on the
*same features the GNN sees*. Everything is held identical to the GNN run so the
log losses are directly comparable:

- same data (experiment `ah_group_switching`, doubled = 100 episodes)
- same seed (`38381`) and fold logic (`get_cross_validations`, grouped by `pair_id`)
- features pulled from the **same tensors** `create_torch_data` builds
- same target, same mask, same metric (`sklearn.metrics.log_loss`, multiclass)
- GNN reference = final-epoch test log loss, averaged over the 5 folds

**Lower is better.** The LR (same features) is the bar the GNN's RNN + graph
architecture must clear; the floor is what either must beat to be worth anything.

Repro: `scripts/baselines/{switch_logit_baseline,contribution_baseline,punishment_baseline}.py`

### Switch (`does_switch`, 2 levels, mask `switch_valid`)

| Model | Mean test log loss | Per-fold std |
|---|---|---|
| constant floor | 0.6095 | — |
| LR — GNN features | 0.5604 | 0.031 |
| LR — enriched (+ other-group / gap / self-contrib) | 0.5310 | 0.040 |
| **GNN switch model** | **0.5163** | 0.037 |

The GNN beats the same-feature LR by 0.044, but a flat LR *given the explicit
other-group/gap feature* (0.531) lands within fold-noise of the GNN — the missing
feature, not architecture, is the high-value lever.

### Contribution (`contribution`, 21 levels, mask `contribution_valid`)

| Model | Mean test log loss | Per-fold std |
|---|---|---|
| constant floor | 2.8787 | — |
| LR — GNN features | 2.4505 | 0.083 |
| LR — enriched (+ own/other/gap group mean) | 2.4298 | 0.081 |
| **GNN contribution model** | **1.9897** | 0.053 |

Mirror-image of switch: here the enriched own-/other-group feature barely helps
(2.45 → 2.43) and the GNN still wins by a wide margin (1.99). Conditional
cooperation on peers is a weak signal for contribution; the GNN's edge is the
RNN's memory of the player's own contribution trajectory — an **architecture**
gain, not a missing feature (for switch it was the reverse).

### Punishment (`punishment`, 31 levels, mask `punishment_valid`)

| Model | Mean test log loss | Per-fold std |
|---|---|---|
| constant floor | 1.4113 | — |
| LR — GNN features | 1.3413 | 0.101 |
| **GNN punishment model** | **1.2030** | 0.078 |

The LR barely beats the floor (1.41 → 1.34): with **no group feature** (mirroring
the GNN config) a linear model can't express the group-relative rule, so it
collapses toward the marginal. The GNN claws back more via the RNN/graph but the
ceiling is low — consistent with the report's "add a group feature" recommendation.

---

*Caveat: fold logic is reproduced exactly (seed, `get_cross_validations`,
`pair_id` grouping). If anything consumed Python's `random` between seeding and
the shuffle in the original GNN run (e.g. `wandb.init`), individual fold
membership could differ; the averaged CV estimate is robust to this.*
