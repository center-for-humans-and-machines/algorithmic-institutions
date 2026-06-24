# [ACTIVE] Own-group average contribution as a node feature (M3)

## Goal

Give the contribution artificial-human GNN the **own-group average contribution**
as a per-agent **node feature**, so the model has the peer-*contribution* signal
needed to reproduce human conditional cooperation after a group switch. This is
arm **M3** of the M0–M5 plan in
`reports/expressiveness_group_switching_contribution_50ep.md` §8 and the
follow-up to #112.

#112 added `same_group` as an *edge* feature (M1) and found **no significant
impact** (test log-loss 1.9968 vs M0 1.9892; sim Q4 post-switch peer β ≈ +0.01
vs human +0.18) — because `same_group` only tells the graph the group
*structure*, not the contribution *level* its peers are at. M3 hands the
aggregate **directly** as a node feature: the cleaner, more-learnable version,
and the arm the report expects to actually move behaviour. M4 (this feature
**plus** `same_group`) is the natural combined test — `same_group` finally has a
real signal to route.

**The feature**, per agent at round `t`: the **leave-one-out mean of the agent's
CURRENT group's members' PREVIOUS-round (`t-1`) contribution**. A per-agent
scalar → a node feature in `x_encoding` (like `prev_contribution`), in contrast
to `same_group`'s per-edge feature.

Scope is the node feature + M3 config only. Significance testing
(paired / multi-seed) is deferred to #115.

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | `parse_agent_rounds` (`data.py`) | Compute the LOO own-current-group prev-round contribution mean as a new float column, directly (not via the `prev_` shift machinery) | |
| 2 | `data_names` + `get_default_values` (`data.py`) | Register the new column dtype (`th.float`) and its round-0 / boundary default | |
| 3 | New M3 config | Sibling of `group_switching_contribution_50ep.yml`, adds the node feature to `x_encoding` (`etype: float`, `norm: 20`), separate `output_dir` | |
| 4 | New M4 config | Same as M3 plus `edge_encoding: [same_group]` (combined arm) | optional |
| 5 | Backward compatibility | New column is just an available feature; old configs/models unaffected | |
| 6 | Tests | `src/aimanager/tests/test_own_group_feature.py` — LOO correctness incl. mid-episode switch, round-0 default, encode/shape wiring | |
| 7 | Verify | Sim Q4 behavioural check (primary) + M3-vs-M0 CV log-loss (secondary, single run) | |

### 1. New column in `parse_agent_rounds`

- **Where:** `src/aimanager/generic/data.py`, `parse_agent_rounds`
  (after `agent_group` is set, line 49; the frame is already sorted by
  `["episode_id", "player_id", "round_number"]` at line 54 — reuse that sort).
- **Column name:** `own_grp_prev_mean_c` (matches
  `human_behavior_analysis.py:75`, so the feature name is consistent across the
  codebase and the verification scripts).
- **Statistic** — per `(episode_id, round_number, group_id)` over the CURRENT
  group's members, the leave-one-out mean of each member's PREVIOUS-round
  contribution. Concrete pandas (after the existing sort):
  ```
  # each player's own previous-round contribution (the t-1 lag)
  prev_c = df.groupby(["episode_id", "player_id"])["contribution"].shift(1)
  # sum / count of prev_c over the CURRENT group at round t
  key = ["episode_id", "round_number", "group_id"]   # group_id == current group
  gsum = prev_c.groupby([df[k] for k in key]).transform("sum")
  gcnt = prev_c.groupby([df[k] for k in key]).transform("count")
  # leave-one-out: exclude self's own prev_c
  loo = (gsum - prev_c) / (gcnt - 1)
  df["own_grp_prev_mean_c"] = loo.astype(float)
  ```
  Note: `group_id` is still present at this point (it is dropped only at line
  88); `agent_group` is its int alias. Use whichever is in scope — they are
  equal. Because the grouping key is the round-`t` group, members are the
  agent's CURRENT group, while `prev_c` carries each member's round `t-1`
  contribution. **This is exactly the combination the `prev_` machinery cannot
  produce** (see below).
- **No target leakage:** the LOO mean is built from `shift(1)` contributions
  (round `t-1`), never round `t` (the contribution being predicted). This
  mirrors how `prev_contribution` is the model's existing self-signal. Confirm
  by inspection: `contribution` at round `t` never enters `own_grp_prev_mean_c`
  at round `t`.

#### The shift-then-feature trap (why we compute directly, not via `prev_`)

`create_torch_data_new` auto-creates a `prev_<k>` tensor for **every** key in
`default_values` by `shift()`-ing along the round axis
(`data.py:158-165`). A naive approach — add a *current-round* own-group mean
column and let that machinery shift it — gives the **WRONG** value at a switch:
the shifted column carries the round `t-1` group's mean (the OLD group's
composition), whereas M3 wants the round-`t` (CURRENT/NEW) group's members'
round `t-1` contributions. Concretely, for an agent that switches into group B
at round `t`:
- shift-of-current-mean → B-at-`t-1`? No — it is the agent's group **at `t-1`**
  (group A) mean — wrong group entirely.
- M3 target → group B members' contributions **at `t-1`** — what the agent's new
  peers were doing before it arrived.

So the feature **must** be computed directly in `parse_agent_rounds`, fully
formed (current-group membership × previous-round contribution baked in), and
then placed into the tensor as a plain per-`(episode, round, agent)` value. It
must **not** be consumed through its auto-generated `prev_own_grp_prev_mean_c`
shift — the model's `x_encoding` references `own_grp_prev_mean_c` directly. The
plan states this explicitly so the engineer does not reuse the `prev_` path.

#### Leave-one-out and invalid contributions

- **Leave-one-out (exclude self):** `(gsum - prev_c) / (gcnt - 1)`, matching
  `human_behavior_analysis.py:71` and `sim_q4_adaptation.py:65`. The mean
  describes the agent's *peers*, so self is excluded.
- **Invalid contributions (`contribution_valid == False`, i.e.
  `player_no_input != 0`):** at line 46 missing contributions are
  `fillna(0)`-ed, so a no-input member would otherwise contribute a spurious 0
  to the peer mean. **Recommendation: exclude invalid contributions from both
  the sum and the count**, so the mean reflects only peers who actually acted.
  Rationale: a timed-out 0 is not a real cooperation signal, and the reference
  analyses operate on the human-action distribution. Implementation: mask
  `prev_c` by the member's round `t-1` `contribution_valid` before the
  groupby-sum/count (e.g. set invalid `prev_c` to NaN and use pandas `sum`/
  `count`, which skip NaN), or compute valid-only sum and count separately.
  **This is a DECISION flagged for human review** (see Decisions) — the
  alternative (count timed-out 0s) is defensible if we want the feature to match
  what a real participant would observe on screen.

### 2. `data_names` dtype + default value

- **Where:** `src/aimanager/generic/data.py`.
- **`data_names`** (`create_torch_data_new`, lines 126-139): add
  `"own_grp_prev_mean_c": th.float`. This makes the column a float tensor so
  `FloatEncoder` accepts it without a dtype error.
- **`get_default_values`** (lines 103-119): add a `c_def` default for round 0 /
  boundary cells — **inherited from the contribution default, not a new choice**
  (see Decisions #2). Round 0 has no previous round, so the per-player `shift(1)`
  lag is NaN there; fill those round-0 `prev_c` cells with `c_def` (mirroring how
  `prev_contribution`'s round-0 cell is already `c_def` via `shift()`), so the
  LOO peer mean evaluates to `c_def` at round 0 by construction. Also set the
  `default_values["own_grp_prev_mean_c"]` tensor-fill entry to `c_def` for cells
  absent from `df`. Implementation: `fillna(c_def)` on the computed column in
  `parse_agent_rounds`. Avoid 0 (biases toward defection at round 0).
- **Do NOT add `own_grp_prev_mean_c` to the set that auto-generates a `prev_`
  shift if that shift would be mistaken for the feature.** It is harmless if the
  `prev_own_grp_prev_mean_c` tensor is created but unused; the config references
  the un-prefixed name. Just confirm the config's `x_encoding` uses
  `own_grp_prev_mean_c`, not the prefixed variant.

### 3. New M3 config

- **New file** (one-config-per-variant, matching how #112 added
  `..._same_group.yml`):
  `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_own_group.yml`.
- **Copy** `group_switching_contribution_50ep.yml` and change only:
  - Add to `model_args.x_encoding`:
    ```
    - name: own_grp_prev_mean_c
      etype: float
      norm: 20
    ```
    `etype: float` dispatches to `FloatEncoder(norm=20, name=...)`
    (`encoder.py:68-80`, `get_encoder` at `encoder.py:100`). `norm: 20` because
    contributions are 0–20 (`n_contributions: 21` levels). `FloatEncoder`
    asserts the column dtype is `th.float` — satisfied by the `data_names`
    entry (section 2). Note the M0 node entries use `encoding: numeric` +
    `n_levels` (→ `IntEncoder`); the new entry uses the `etype`/`norm` form
    because the feature is a continuous float, not an integer level index.
  - Distinct `output_dir`:
    `artifacts/artificial_humans/group_switching_contribution_50ep_own_group`
    (preserves the M0 baseline artifact — test log_loss 1.9892).
  - Label `own_group: true` under `params.labels`.
  - Keep `agent_group` as a node feature and `seed: 38381`, 575 epochs, 5-fold,
    same data — everything else identical to M0.
  - `shuffle_features`: optionally add `own_grp_prev_mean_c` for a
    feature-importance read (it is a stored data key, so the existing
    `shuffle_feature` mechanism can permute it — unlike the derived
    `same_group`). Optional, not required for the primary verification.

### 4. New M4 config (combined arm)

- **New file:**
  `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_own_group_same_group.yml`.
- Identical to the M3 config (section 3) plus
  `model_args.edge_encoding: [{name: same_group, etype: bool}]` and a distinct
  `output_dir` (`..._own_group_same_group`). `same_group` is already supported on
  `main` (merged in #112), so no model code change is needed — just the config.
- This is the arm where `same_group` finally has a real own-group contribution
  signal to route; the M0 / M3 / M4 comparison isolates the node feature alone
  vs. node feature + edge routing.

### 5. Backward compatibility

- The new column is just another available data key. **Only configs that list
  `own_grp_prev_mean_c` in `x_encoding` use it**; M0 and all other configs are
  unchanged and produce identical tensors for the features they request.
- **Existing trained `.pt` files:** unaffected. The feature lives entirely in
  the data pipeline + config; model `save`/`load` already round-trips
  `x_encoding`, and an old checkpoint simply never references the new column.
  No `graph.py` `save`/`load` change is needed for M3 (unlike #112, which added
  a new `edge_encoding` constructor arg). Confirm by inspection that loading an
  M0 `.pt` does not touch `own_grp_prev_mean_c`.
- Adding a key to `data_names` enlarges the in-memory `data` dict by one tensor;
  downstream code iterates by name and is indifferent to the extra key.

### 6. Tests

- **New file** `src/aimanager/tests/test_own_group_feature.py` (run on Raven via
  `scripts/remote_test.sh`; PyG is Linux-only). Mirror the structure of
  `test_edge_encoder.py`.
- **(a) LOO own-group prev-mean correctness, WITH a mid-episode switch (the key
  case):** hand-build a small `df` (one episode, e.g. 4 players, 3 rounds) with a
  player switching groups mid-episode, run `parse_agent_rounds`, and assert
  `own_grp_prev_mean_c`:
  - For a non-switching agent at round `t`: equals the LOO mean of its
    co-members' round `t-1` contributions, self excluded.
  - **For the switched agent at the arrival round `t`: equals the LOO mean of
    its NEW group's members' round `t-1` contributions — NOT the old group's.**
    Construct contributions so the two groups have clearly different prev means
    (e.g. old group prev mean 2, new group prev mean 15) and assert the value is
    the new-group number. This is the assertion that distinguishes M3 from the
    naive shift-trap implementation.
  - Concrete: build rounds so that, for the switcher, `gsum`/`gcnt` over the new
    group at round `t` minus self gives a known constant; assert
    `pytest.approx`.
- **(b) Round-0 default:** assert every agent's `own_grp_prev_mean_c` at round 0
  equals the chosen default (`c_def`), since there is no previous round.
- **(c) Invalid-contribution handling:** include a member with
  `contribution_valid == False` at round `t-1` and assert it is excluded from
  the peer mean (matches the section-1 recommendation). If the human-review
  decision flips to "include", update this assertion accordingly.
- **(d) Encode / shape wiring:** build a `GraphNetwork` with an `x_encoding`
  containing the new float entry, run `encode` on a small `data` dict, and assert
  the encoded `x` width grew by exactly 1 and the column equals
  `own_grp_prev_mean_c / 20`. Confirms `FloatEncoder` wiring and dtype
  (mirror the encode tests in `test_edge_encoder.py` / `test_encoder.py`).
- Run the full suite to confirm no regression in environment/manager paths.

### 7. Verification (single run is fine; significance deferred to #115)

- Train **both M3 and M4** on Raven (seed 38381, 5-fold), then a sim twin per
  arm (mirroring `19_2g8a_rule_based_vs_zero_same_group.yml`).
- **Primary — sim behaviour (Q4):** run `scripts/data_analysis/sim_q4_adaptation.py`
  on the M3 and M4 sims. Question: does the post-switch new-group peer β rise
  toward the human **+0.18** (vs ≈ +0.01 for M0/M1)? This is the meaningful
  behavioural signal for a peer-contribution feature; `new_peers` is the new
  group's LOO mean at arrival — the target M3 is meant to let the AH track. M4
  shows whether adding `same_group` routing on top helps the model use it.
- **Secondary — M0 / M3 / M4 CV test log-loss** (single run, vs M0 **1.9892**)
  as a sanity read. The report's prior expected effect (~0.02) is near the
  single-run noise floor (~0.014, per #112), so **do not over-read** the deltas —
  a proper paired / multi-seed test is #115.

## Implementation notes

- The existing sort in `parse_agent_rounds` (line 54,
  `["episode_id", "player_id", "round_number"]`) is the correct order for the
  `groupby([...])["contribution"].shift(1)` per-player lag. Compute the lag
  first, then the per-`(episode, round, group)` aggregate — do not re-sort
  between the two.
- For the invalid-exclusion path, the cleanest pandas is to NaN out invalid
  `prev_c` (`prev_c = prev_c.where(prev_valid)`), then `groupby.transform("sum")`
  and `transform("count")` (both skip NaN), giving valid-only sum/count; then
  LOO-subtract the agent's own *valid* `prev_c` (or skip the subtraction when the
  agent's own prev is invalid — guard the `gcnt - 1` denominator against 0/NaN
  and fall back to the default).
- Guard `gcnt - 1 == 0` (a group with a single valid member at `t-1`): fall back
  to the round-0 default rather than dividing by zero.
- `FloatEncoder` (`encoder.py:75-80`) divides by `norm` and `unsqueeze(-1)`s — it
  needs a `th.float` tensor; the `data_names` entry guarantees this. No decode
  path is needed (it is an input feature, never a target).

## Decisions (resolved with user)

1. **Exclude invalid (`contribution_valid == False`) contributions from the peer
   mean** — yes. A timed-out 0 is not a cooperation signal.
2. **Round-0 / boundary default is inherited, not chosen.** It must equal the
   contribution default `c_def`, because round-0 `prev_contribution` is *already*
   `c_def` for every agent (`shift()` sets `tensor[:, :, 0] = default` with
   `default = default_values["contribution"] = c_def`; verified at `data.py:99`,
   `:105`, `:158-165`). Building the peer mean from those defaulted previous
   contributions makes round 0 = mean-of-`c_def`s = `c_def` automatically — so
   don't invent a parallel default; reuse the existing one (and set the
   `default_values` tensor-fill entry to `c_def` for consistency).
3. **M4 combined config (M3 + `same_group`): included in this PR.** Build the
   combined arm alongside M3 so `same_group` (from #112) finally has a real
   own-group contribution signal to route — the comparison M0 / M3 / M4 is the
   point. Section 4.

## Next Actions

- [ ] Add `own_grp_prev_mean_c` to `parse_agent_rounds` (section 1), computed
      directly (current-group membership × `shift(1)` contribution, LOO,
      invalid-excluded, round-0 filled with `c_def`).
- [ ] Add the `data_names` (`th.float`) and `get_default_values` entries
      (section 2); confirm the config references the un-prefixed name.
- [ ] Add the M3 config `group_switching_contribution_50ep_own_group.yml`
      (section 3), separate `output_dir`, M0 preserved.
- [ ] Add the M4 combined config
      `group_switching_contribution_50ep_own_group_same_group.yml` (section 4),
      separate `output_dir`.
- [ ] Add `src/aimanager/tests/test_own_group_feature.py` (section 6); run on
      Raven via `scripts/remote_test.sh`.
- [ ] Train M3 on Raven; run sim Q4 (`sim_q4_adaptation.py`) as the primary
      check and report M3-vs-M0 CV log-loss as a secondary sanity read
      (section 7).
