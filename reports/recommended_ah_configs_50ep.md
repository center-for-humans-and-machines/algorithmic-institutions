# Best-guess configurations for the three AH models (50ep)

Synthesises the three analyses on this PR — feature expressiveness
(`expressiveness_*`), human-behaviour drivers (`human_behavior_analysis_50ep.md`),
and the overfitting/regularization diagnosis
(`regularization_recommendation_contribution_ah.md`) — into one proposed config
per model.

**These are hypotheses, not tuned results — validate each with 5-fold CV log-loss
against the current config.** They **assume the key features are implemented**:

- `same_group` **edge** feature — `1[group_src == group_dst]` (see the contribution
  report §8; the edge path is already plumbed).
- a `common_good_gap` (= other-group − own-group) and/or `other_group_common_good`
  preprocessing column.
- the punishment model can read the **current-round** `contribution` (the AH stack
  already predicts contributions before punishment within a round).

## Shared training recipe (all three)

From the overfitting diagnosis — edge + RNN earns the gains but the GRU overfits:

- **`Adam → AdamW`** so weight decay is decoupled and actually regularizes.
- **Weight decay is the primary regularizer** (ideally a higher-decay param group on
  the GRU); **keep early stopping** as the free backstop (`early_stopping_patience`,
  already supported in `train.py`).
- **Do not increase `hidden_size`** (the problem is variance, not capacity); a
  smaller GRU is the only capacity knob worth trying, never dropout (single-layer
  width-20 GRU).
- Keep `add_edge_model: True`, `add_rnn: True` — both do real work; node-only
  underfits badly.

---

## 1. Contribution AH

Behaviour: dominated by self-inertia (β +0.69) + **own-group** conditional
cooperation (+0.17); the **other group is ignored** (+0.02). So the only structural
need is to route the *own*-group social signal cleanly.

```yaml
params:
  mask_name: contribution_valid
  switch_every: 4
  experiment_names: [ah_group_switching]
  n_cross_val: 5
  n_player: 8
  n_groups: 2
  data_file: experiments/2group_8agent_50ep.csv
  model_name: graph
  model_args:
    y_levels: 21
    y_name: contribution
    hidden_size: 20            # do NOT increase; optionally test 15
    add_rnn: True
    add_edge_model: True
    add_global_model: False
    same_group_edge: True      # KEY — clean own-vs-other message routing
    x_encoding:
      - {name: prev_contribution, n_levels: 21, encoding: numeric}
      - {name: prev_punishment,  n_levels: 31, encoding: numeric}
      # agent_group DROPPED: unused (shuffle Δ +0.0009) and subsumed by same_group
  optimizer_args:
    optimizer: adamw           # decoupled weight decay
    lr: 3.e-4
    weight_decay: 3.e-4        # up from 1e-5 — edge+rnn overfits hard
  train_args:
    epochs: 1500
    batch_size: 4
    clamp_grad: 1
    eval_period: 25
    early_stopping_patience: 200
    l1_entropy: 0
  device: cuda
  autoregression: False
```

| Choice | Evidence |
|---|---|
| keep `prev_contribution` | top driver, β +0.69 |
| `same_group` edge (not `agent_group`) | graph then computes own-group mean of neighbours' `prev_contribution` = conditional cooperation (+0.17); `agent_group` was unused |
| no other-group feature | other-group effect ≈ 0 (+0.02) |
| AdamW + wd 3e-4 + early stop | the overfitting diagnostic (RNN-driven) |
| (optional) add own-group mean contribution node feature | the +0.027 R² direct signal — belt-and-suspenders if `same_group` alone underperforms |

---

## 2. Switch predictor AH

Behaviour: switching **is a between-group comparison** — the **gap** (other − own
common good) is the dominant driver (+0.57 log-odds/SD); punishment flight (+0.46);
settling over time (−0.20). The current config has *own*-group common good but only
reaches the other group through graph-gating it doesn't learn → give it the
comparison directly.

```yaml
params:
  mask_name: switch_valid
  switch_every: 4
  experiment_names: [ah_group_switching]
  n_cross_val: 5
  n_player: 8
  n_groups: 2
  data_file: experiments/2group_8agent_50ep.csv
  model_name: graph
  model_args:
    y_levels: 2
    y_name: does_switch
    hidden_size: 10
    add_rnn: True
    add_edge_model: True
    add_global_model: False
    same_group_edge: True            # lets the graph carry the OTHER group's quality
    x_encoding:
      - {etype: float, name: prev_common_good, norm: 20}         # own-group quality (β -0.40)
      - {etype: float, name: prev_common_good_gap, norm: 20}     # KEY — other-own (β +0.57)
      - {name: prev_punishment, n_levels: 31, encoding: numeric} # punishment flight (β +0.46)
      - {name: round_number,   n_levels: 24, encoding: numeric}  # commitment over time (β -0.20)
      # prev_agent_group DROPPED: unused (shuffle Δ +0.0005)
  optimizer_args:
    optimizer: adamw
    lr: 5.e-4
    weight_decay: 1.e-3              # already well-regularized; keep
  train_args:
    epochs: 600
    batch_size: 10
    eval_period: 10
    early_stopping_patience: 100
    l1_entropy: 0
  device: cuda
  autoregression: False
```

| Choice | Evidence |
|---|---|
| add `prev_common_good_gap` | the decision rule — gap is the single best predictor (β +0.57) |
| `same_group` edge | complementary route to the other group's quality via the graph |
| keep `prev_common_good` | own-group quality; top feature (shuffle Δ +0.111) |
| keep `prev_punishment`, `round_number` | flee punishment (+0.46); settle late (−0.20) |
| drop `prev_agent_group` | unused (Δ +0.0005); also fixes its spurious numeric encoding |

---

## 3. Punishment AH

Behaviour: a **group-relative** rule — punish low contributors (β −0.34) *relative*
to the own-group mean (+0.21); persistence (+0.40); end-game easing (−0.09). Two
gaps in the current config: it has **no group feature** (can't compute the relative
mean) and it reads **lagged** contribution, while managers react to the
**current** round (r −0.28 vs −0.19).

```yaml
params:
  mask_name: punishment_valid
  switch_every: 4
  experiment_names: [ah_group_switching]
  n_cross_val: 5
  n_player: 8
  n_groups: 2
  data_file: experiments/2group_8agent_50ep.csv
  model_name: graph
  model_args:
    y_levels: 31
    y_name: punishment
    hidden_size: 20
    add_rnn: True
    add_edge_model: True
    add_global_model: False
    same_group_edge: True            # KEY — graph computes the OWN-group mean for the relative rule
    x_encoding:
      - {name: contribution,    n_levels: 21, encoding: numeric}  # KEY — CURRENT round, not prev (timing)
      - {name: prev_punishment, n_levels: 31, encoding: numeric}  # persistence (β +0.40)
      - {name: round_number,    n_levels: 24, encoding: numeric}  # end-game easing (β -0.09)
      - {etype: bool, name: is_first}
  optimizer_args:
    optimizer: adamw
    lr: 1.e-4
    weight_decay: 1.e-4              # up from 1e-5; epochs were very high (1250)
  train_args:
    epochs: 1000
    batch_size: 10
    clamp_grad: 1
    eval_period: 10
    early_stopping_patience: 150
    l1_entropy: 0
  device: cuda
  autoregression: False
```

| Choice | Evidence |
|---|---|
| `contribution` (current, not `prev_contribution`) | manager reacts to current round (r −0.28 vs −0.19); stack provides it |
| `same_group` edge | enables own-group mean of current contributions = the relative rule (+0.21) |
| keep `prev_punishment` | strong persistence (β +0.40) |
| add `round_number` | end-game easing (β −0.09) |
| AdamW + wd 1e-4 + early stop | 1250-epoch training is over-long; regularize |

> If within-round coordination of punishments matters, the autoregressive variant
> (predict agents in order) is the fallback — but the simplified non-autoregressive
> model is the better default to start from.

---

## One-line summary per model

| Model | The one change that matters most | Supporting changes |
|---|---|---|
| Contribution | `same_group` edge (clean conditional cooperation) | drop `agent_group`; AdamW + wd 3e-4 + early stop |
| Switch | add the **gap** (other − own common good) | `same_group` edge; drop `prev_agent_group` |
| Punishment | feed **current-round** contribution + `same_group` edge (relative rule) | add `round_number`; AdamW + wd + early stop |

**Cross-cutting theme:** all three behaviours are *reference-dependent on the own
group*, so the highest-leverage change everywhere is making the **own-vs-other group
distinction explicit** (`same_group` edge / `gap` feature) instead of asking the
graph to learn it — exactly the variable assumed implemented here.

## Caveats

- Best-guesses from linear/associational evidence; confirm each with 5-fold CV and a
  feature-shuffle importance check before adopting.
- The regularization values (`weight_decay`, `epochs`, patience) are starting points
  for a short sweep, not final.
- `same_group`, `common_good_gap`, and current-round `contribution` plumbing must be
  implemented first (assumed here).
