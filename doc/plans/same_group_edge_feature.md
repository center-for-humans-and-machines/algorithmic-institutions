# [ACTIVE] same_group edge feature for the contribution AH GNN

## Goal

Give the contribution artificial-human GNN an explicit per-edge `same_group` bit
so the edge MLP can tell own-group neighbours from other-group neighbours
*directly*, instead of having to infer the equality `g_src == g_dst` from the two
endpoints' `agent_group` one-hots inside a single `linear+tanh` unit (see
`reports/expressiveness_group_switching_contribution_50ep.md` §4, caveat c, and
§5b: `agent_group` importance ≈ 0). `same_group(i,j) = 1` iff agents `i` and `j`
share a sub-group in that round, else `0`, computed per decision round (groups
switch over time).

Scope is the **edge feature only**. The own-group average-contribution *node*
feature (report §7, M3) is a separate issue. The encoding mechanism added here
must be general enough to host future edge features, but only `same_group` is
wired and tested now.

## Plan

| # | Section | Change | Optional |
|---|---|---|---|
| 1 | `GraphNetwork.__init__` | New `edge_encoding` config section → build a derived edge encoder, replace `EmptyEncoder`; size the `EdgeModel` from it | |
| 2 | New `SameGroupEdgeEncoder` (in `graph.py`) | Derived edge encoder: consumes `agent_group` + `edge_index`, emits `(E, n_rounds, F_e)`; default empty when no `edge_encoding` | |
| 3 | `GraphNetwork.encode` | Populate `edge_attr` via the edge encoder after `edge_index` is built | |
| 4 | `GraphNetwork.forward` | No change needed (already reads `edge_attr` if present); verify the empty fallback still fires when off | |
| 5 | `save` / `load` | Add `edge_encoding` to the `to_save` list | |
| 6 | Config | Add `edge_encoding: [{name: same_group, etype: bool}]` to `group_switching_contribution_50ep.yml`; keep `agent_group` node feature | |
| 7 | Tests | `src/aimanager/tests/` — same_group correctness, shapes, backward-compat (run on Raven) | |
| 8 | Verify / ablate | Retrain (M1 arm), compare CV test log-loss against M0 baseline (1.9897) and LR (2.4505) | |

### 1. `edge_encoding` config section and `__init__` threading

- **Where:** `src/aimanager/generic/graph.py`, `GraphNetwork.__init__`
  (lines 82-116, 125-133).
- Add a new keyword `edge_encoding=[]` to the signature (alongside `x_encoding`,
  `u_encoding`). It is a list of encoding dicts, same shape as `x_encoding`, e.g.
  `[{name: same_group, etype: bool}]`.
- Replace the hardcoded `self.edge_encoder = EmptyEncoder(refrence=y_name)`
  (line 111) with a constructed edge encoder driven by `edge_encoding`
  (section 2). When `edge_encoding` is empty/absent, the encoder reports
  `size = 0` and emits the empty tensor — preserving today's behaviour.
- Store `self.edge_encoding = edge_encoding` as an attribute (mirrors
  `self.x_encoding` at line 117) so it can be persisted (section 5).
- `edge_features = self.edge_encoder.size` (line 116) already feeds
  `EdgeModel(edge_features=…)` at line 129 — with `size = 1` the edge MLP input
  width auto-grows from `2*x_features + 0 + u_features` to
  `2*x_features + 1 + u_features`. **No `EdgeModel` change.**

### 2. A derived edge encoder (new class in `graph.py`)

- **Where:** new class in `src/aimanager/generic/graph.py`, next to
  `EmptyEncoder` (lines 71-79).
- **Why a new class, not `encoder.py`:** the per-node `Encoder` /
  `IntEncoder` / `BoolEncoder` operate on a single named state tensor of shape
  `(…, n)` and append a feature axis. An edge feature like `same_group` is
  *relational* — it depends on **two** node tensors selected through
  `edge_index` (`agent_group[row]` vs `agent_group[col]`), which `encoder.py`'s
  `forward(**state)` signature cannot express. Keeping it in `graph.py` next to
  `EmptyEncoder` (which already takes `n_edges` / `edge_index`-adjacent context)
  keeps `encoder.py` purely per-node. Justification recorded here so the engineer
  does not try to force it into `Encoder`.
- **Design — `EdgeEncoder` dispatch + `SameGroupEdgeEncoder`:**
  - A small `EdgeEncoder` wrapper holds a `ModuleList` of per-feature edge
    encoders built from `edge_encoding` (parallels `Encoder`), summing their
    `.size`. With an empty list it behaves exactly like `EmptyEncoder`
    (`size = 0`, returns the empty tensor) — this is the backward-compat default.
  - `SameGroupEdgeEncoder(name="same_group", etype="bool")` has `size = 1` and a
    `forward(self, *, edge_index, n_rounds, **state)` that computes the bit.
  - Dispatch by `name`: a tiny registry `{"same_group": SameGroupEdgeEncoder}`.
    A new future edge feature adds one class + one registry entry. The `etype`
    key is accepted for symmetry with node encodings (and validates that
    `same_group` is `bool`), but the relational compute lives in the class.
- **`forward` signature:** the edge encoder must receive `edge_index` and the
  round count, plus the flattened state dict. It returns `(E, n_rounds, F_e)`,
  `float`.

### 3. Computing the `same_group` edge_attr (shapes)

- **Where:** `src/aimanager/generic/graph.py`, `GraphNetwork.encode`
  (lines 234-271), after `edge_index` is set (line 269).
- **Input layout (verified):** in `encode`, `data` tensors are
  `(n_batch, n_player, n_rounds)`. Node encoders produce
  `(n_batch, n_player, n_rounds, F)`, then `flatten(0, 1)` →
  `(N, n_rounds, F)` with `N = n_batch * n_player` (lines 262-263). `edge_index`
  from `create_fully_connected` (lines 388-398) uses node ids offset per batch
  element (`i + k*n_player`), so its entries index directly into that flattened
  `N` dimension. The edge feature must therefore be built **on the flattened,
  batch-offset node tensor** so batching is automatic.
- **Concrete steps for `same_group`:**
  ```
  ag    = data["agent_group"].flatten(0, 1)          # (N, n_rounds)  int64
  row, col = edge_index                              # each (E,)
  same  = (ag[row] == ag[col])                       # (E, n_rounds)  bool
  edge_attr = same.float().unsqueeze(-1)             # (E, n_rounds, 1)
  ```
  `ag[row]` / `ag[col]` are gathers over dim 0 → `(E, n_rounds)`. Because
  `edge_index` already carries the per-batch offset, edges never cross batch
  boundaries, so no separate batch handling is needed.
- **Round dimension:** `agent_group` is time-varying (refreshed at switches), so
  indexing the full `(N, n_rounds)` tensor yields a per-round `same_group`
  automatically — matching how the round axis is carried for node features and
  the empty `edge_attr` fallback in `forward` (`x.shape[1]` = `n_rounds`,
  lines 217-218).
- The edge encoder is invoked here as
  `self.edge_encoder(**encoded_state, edge_index=edge_index, n_rounds=n_rounds)`
  and the result stored as `encoded["edge_attr"]`. Pass the *flattened* node
  tensors (or `agent_group` specifically) so `ag` has the `(N, n_rounds)` layout
  above. **`encode` is the single chokepoint** for training,
  `predict_independent`, and `predict_autoreg` — so this one insertion covers
  every path.
- **Device:** build on `edge_index.device` (already moved to `device` at
  line 270; ensure `edge_attr` is included in that `.to(device)` sweep or built
  on-device).

### 4. `forward` — verify, do not change

- **Where:** `GraphNetwork.forward`, lines 211-232. It already does
  `if "edge_attr" in data: edge_attr = data["edge_attr"]` else builds the empty
  `(E, n_rounds, 0)` tensor. With the feature on, `encode` supplies a non-empty
  `edge_attr` that flows untouched into `op1` → `EdgeModel.forward` (line 17-24).
- **Action:** confirm no edit is required; the only risk is a width mismatch
  between `edge_attr.shape[-1]` and the `EdgeModel` linear's expected
  `edge_features` — guaranteed consistent because both derive from
  `self.edge_encoder.size`.

### 5. save / load

- **Where:** `GraphNetwork.save`, `to_save` list (lines 360-374).
- Add `"edge_encoding"` to `to_save` so a trained model reloads with the same
  edge feature set. `load` (lines 377-382) passes `**to_load` into `__init__`,
  which rebuilds the edge encoder from `edge_encoding`; the trained `EdgeModel`
  weights live inside `op1` (already saved wholesale), so the sized linear
  survives reload — only the flag/list is needed to re-enable construction.

### 6. Config change

- **Where:** `configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml`,
  under `model_args` (after `x_encoding`, lines 48-57).
- Add:
  ```yaml
  edge_encoding:
    - name: same_group
      etype: bool
  ```
- **Keep `agent_group` as a node feature (default).** Report §8 argues it is
  likely droppable once `same_group` exists (M5 arm), but defaults to keeping it;
  the drop is a separate ablation arm, not part of this change.

### 7. Backward compatibility

- **Existing configs without `edge_encoding`:** `edge_encoding=[]` default →
  `EdgeEncoder` with empty `ModuleList` → `size = 0` and the empty
  `(…, 0)` tensor, identical to today's `EmptyEncoder`. `EdgeModel` input width
  is `2*x_features + 0 + u_features` as before.
- **Existing trained `.pt` files (saved before this change):** their `to_load`
  dict has no `edge_encoding` key; `__init__` must default it to `[]` (the
  signature default handles this), so they load and run with empty `edge_attr`.
  Verify `load` does not error on the missing key.
- **`forward` fallback** (section 4) covers any path where `edge_attr` is absent
  from the data dict.

### 8. Verification / ablation

- This is the **M1 arm** of the M0–M5 plan in
  `reports/expressiveness_group_switching_contribution_50ep.md` §6 (M1 = baseline
  + `same_group` edge feature).
- Retrain with the edited config (575 epochs, 5 folds, seed 38381) on Raven and
  compare CV-averaged final-epoch **test log-loss** against:
  - **M0 GNN contribution baseline: 1.9897** (`reports/ah_baseline_log_loss.md`).
  - **LR same-features floor: 2.4505** (already cleared by M0).
- **Prior (report §5a / §7):** other-group contribution effect ≈ 0 and own-group
  β ≈ 0.19, so M1's expected log-loss improvement is *small*; the value of M1 is
  enabling clean own/other routing for when the own-group node feature (M3)
  lands, more than a large standalone gain. Report the delta either way; the CV
  number is the verdict.

## Implementation notes

- The edge encoder's `forward` needs `edge_index` and `n_rounds`, which are not
  in `state`. Pass them as explicit kwargs from `encode` (mirror how
  `EmptyEncoder.forward(*, n_edges, **state)` takes extra kwargs). Drop the old
  `EmptyEncoder` once `EdgeEncoder` covers the empty case, or keep it as the
  empty-list internal — engineer's choice, but avoid two parallel empty paths.
- Keep `same_group` strictly derived (no new stored column in `data.py`):
  `agent_group` already exists as an int64 node tensor (data.py:32, 49, 135).
- Watch the `encode` device sweep (line 270): it filters `if v is not None`.
  Ensure `edge_attr` is non-None and included.

## Decisions (resolved with user)

- **Encoder placement:** in `graph.py` (relational, can't fit `encoder.py`'s
  per-node `forward`). Confirmed.
- **Edge encoding:** single `bool` 0/1 per edge (`etype: bool`), per section 2.
- **Ablation scope:** wire `same_group` only and **keep `agent_group`** as a node
  feature; the M5 `agent_group`-drop arm is a separate follow-up.
- **Status:** `[ACTIVE]` — implementation in progress on branch
  `112-same-group-edge-feature`.

## Next Actions

- [x] Add `EdgeEncoder` + `SameGroupEdgeEncoder` to `graph.py` (section 2).
- [x] Thread `edge_encoding` through `__init__`, replacing `EmptyEncoder`
      (section 1). `EmptyEncoder` removed; empty `edge_encoding` keeps prior
      behaviour.
- [ ] Populate `edge_attr` in `encode` (section 3); verify `forward` fallback
      (section 4).
- [ ] Add `edge_encoding` to `save` `to_save` list (section 5).
- [ ] Edit `group_switching_contribution_50ep.yml` (section 6).
- [ ] Add unit tests in `src/aimanager/tests/` (section 7): (a) `same_group`
      correctness for a hand-built 2-group assignment incl. a mid-episode switch;
      (b) `edge_attr` shape `(E, n_rounds, 1)`; (c) backward-compat: no
      `edge_encoding` → empty `(E, n_rounds, 0)` edge_attr and model runs; (d)
      save/load round-trips `edge_encoding`.
- [ ] Run tests on Raven: `scripts/remote_test.sh -- -k edge -v`.
- [ ] Retrain M1 on Raven; record CV test log-loss vs 1.9897 (section 8).
