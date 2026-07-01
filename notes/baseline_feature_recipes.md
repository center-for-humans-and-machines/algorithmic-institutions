# How each of the 30 features is computed

A step-by-step recipe for reproducing every feature value for one agent across a
whole episode by hand, using the raw per-round tables in `feature_verification.md`
(`agent_group`, `contribution`, `punishment`, `common_good`, `does_switch`).
Implemented in `scripts/baselines/handcrafted_grid.py :: build_feature_pool`.

---

## Conventions & inputs

- **Episode** = one `group_idx`. **Agent** = `a`. **Round** = `t` (0 … T−1).
- **`prev` = the previous round, t−1.** Round 0 has no previous round → uses defaults.
- **Membership for all `prev_*_mean_*` features uses the *previous* round's groups**
  (`prev_agent_group`): who was in which sub-group at t−1. Structural `group_size`
  (no prefix) uses the *current* round's groups.
- **Round-0 / default constants** (dataset medians): `c_def = 9`, `p_def = 0`,
  `cg_def = 13.17`. These seed every `prev_*` value at round 0.
- **Payoff** (per contributor per round, from `reports/basics.md`):
  `payoff = 20 − contribution − punishment + common_good`
  (`common_good` is already the per-capita share).

To trace a feature at round `t`, look at **round `t−1`** in the raw tables (except the
structural block, which reads round `t`).

---

## B1 — self (my own t−1 outcome)

Look up **my own** row at round t−1:
| feature | recipe | round-0 default |
|---|---|---|
| `prev_contribution` | my `contribution` at t−1 | `c_def = 9` |
| `prev_punishment` | my `punishment` at t−1 | `p_def = 0` |
| `prev_common_good` | my `common_good` at t−1 | `cg_def = 13.17` |
| `prev_payoff` | `20 − prev_contribution − prev_punishment + prev_common_good` | `24.17` |

## B2 — own-group peers (leave-one-out, t−1)

1. Find **my group at t−1** (my `agent_group` value in row t−1).
2. **Peers** = the *other* agents who were in that same group at t−1 (exclude me).
3. Average their t−1 values:

| feature | recipe | if I have no peers |
|---|---|---|
| `prev_contribution_mean_peers` | mean of peers' `contribution` at t−1 | 0 |
| `prev_punishment_mean_peers` | mean of peers' `punishment` at t−1 | 0 |
| `prev_payoff_mean_peers` | `20 − (peers' mean contribution) − (peers' mean punishment) + (peers' mean common_good)` | — |

*(The peers' mean common_good used by payoff isn't exposed as its own feature — B2
drops it because common_good is group-level, so it already equals `prev_common_good`.)*

## B3 — other group (t−1)

1. The **other sub-group at t−1** = agents whose t−1 group ≠ my t−1 group.
2. Average over **all** its members (not leave-one-out — I'm not in it):

| feature | recipe |
|---|---|
| `prev_contribution_mean_other` | mean of other group's `contribution` at t−1 |
| `prev_punishment_mean_other` | mean of other group's `punishment` at t−1 |
| `prev_common_good_mean_other` | the other group's t−1 `common_good` |
| `prev_payoff_mean_other` | `20 − other_c − other_p + other_cg` |

**Empty other group** (everyone merged into one sub-group at t−1): treat it as a
0-sized, all-zero group → every `_mean_other` = **0**.

## B4 — gap (own − other, t−1)

Simply subtract B3 from B2, per measure:

`prev_{m}_mean_gap = prev_{m}_mean_peers − prev_{m}_mean_other`   for m ∈ {contribution, punishment, common_good, payoff}

*(For common_good the "peers" side is the own group's t−1 cg — i.e. my own
`prev_common_good` — since cg is shared within a group.)* When the other group is
empty, gap = peers − 0 = peers.

## B5 — own-group since-switch window

Running mean of the **B2 peer value** over my current tenure (the rounds since I last
switched), inclusive of t:

1. `t0` = the most recent round ≤ t where **my** `does_switch = 1` (or 0 if I never switched).
2. `win_{m}_mean_peers[t] = average of prev_{m}_mean_peers over rounds t0 … t`.
3. `win_payoff_mean_peers = 20 − win_contribution_mean_peers − win_punishment_mean_peers + win_common_good_mean_peers`.

The average **resets** whenever I switch groups.

## B6 — other-group since-switch window

Identical to B5 but averaging the **B3 other-group value** (`prev_{m}_mean_other`)
over the same tenure window.

## B7 — structural / timing (reads round t, not t−1)

| feature | recipe | round-0 |
|---|---|---|
| `round_number` | `t` | 0 |
| `switched_last_choice` | **my** `does_switch` at my most recent *decision round* (r where `r % switch_every == 0`, `r ≠ 0`, `r ≤ t`), carried forward; `switch_every = 4` → decisions at rounds 4, 8, 12, 16, 20 | 0 (no decision yet) |
| `rounds_since_switch` | rounds since I last *actually* switched (`does_switch = 1`); 0 on the switch round | 0 |
| `group_size` | # recorded agents in **my current** group at round t | 4 |
| `prev_group_size` | # agents in my group at t−1 | 4 (balanced start) |
| `prev_group_size_other` | # agents in the other group at t−1 = `total(t−1) − prev_group_size` | 4 |
| `prev_group_size_delta` | `prev_group_size − prev_group_size_other` | 0 |

---

## Worked example — agent a0, round 1 (episode 6)

Look at **round 0** in the raw tables. At round 0, a0's group is 0; group 0 =
{a0, a2, a4, a5}, group 1 = {a1, a3, a6, a7}.

- **B1** `prev_contribution` = a0's round-0 contribution = **16**;
  `prev_punishment` = 0; `prev_common_good` = 15.8;
  `prev_payoff` = 20 − 16 − 0 + 15.8 = **19.8**.
- **B2 peers** = {a2, a4, a5}. Round-0 punishments {0, 0, 1} → `prev_punishment_mean_peers`
  = 1/3 = **0.333**. (contributions {11,12,1} → `prev_contribution_mean_peers` = 8.0.)
- **B3 other** = {a1, a3, a6, a7}. Round-0 punishments {2, 0, 0, 0} →
  `prev_punishment_mean_other` = 2/4 = **0.5**.
- **B4 gap** `prev_punishment_mean_gap` = 0.333 − 0.5 = **−0.167**.
- **B5 window** (no switch before round 1, so tenure = rounds 0–1;
  round-0 peer punishment default = 0): `win_punishment_mean_peers`
  = mean(0, 0.333) = **0.167**.
- **B6 window**: `win_punishment_mean_other` = mean(0, 0.5) = **0.250**.

Note 0.167 and 0.250 both *display* as "0.2" at one decimal in the dump, but they are
different numbers — the table rounds to ±0.05.

---

## Round-0 summary

Because round 0 has no previous round, every `prev_*` / `win_*` value is a default
(see `feature_round0_defaults.md`): self/peers = medians, other = peers (symmetric
kick-off → all gaps 0), sizes = 4/4 (delta 0), `rounds_since_switch = 0`,
`switched_last_choice = 0`. `group_size` is the real (constant 4) round-0 size.
