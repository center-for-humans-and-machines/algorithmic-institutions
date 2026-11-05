# [ACTIVE] Policy finder on experiment 1: rules against the RL GNN manager

Task: `docs/tasks/autoresearch_an_old_policy.md`.

## Why

Experiment 1 deployed a DQN manager: a GNN with an RNN, trained against artificial-human
contribution clones, in one group of 4 players over 24 rounds, without group switching. For
experiment 2 we built the policy finder instead. Agents design interpretable rule-based
managers, and a Sobol sweep, a GP fit and a final run choose the rule's values. The
question here: would the policy finder have matched or beaten the RL manager on
experiment 1's own terms? That means the same clones, the same game, and the metric the RL
manager was trained and judged on: the clones' payoffs.

## What experiment 1 was (found in the repo)

| | Experiment 1 | Experiment 2 (policy finder today) |
|---|---|---|
| Game | 1 group, 4 players, 24 rounds, no switching | 2 groups of up to 8 players, switching every 4 rounds |
| Contribution clone | `21_contribution_model_v4` (`architecture_node+edge+rnn__dataset_full.pt`) | Levin's stack (`24_LEVIN`) |
| Validity (timeout) clone | `22_contribution_valid_model_v4` (`rnn_False__dataset_full.pt`) | `raven_script_22` |
| Clone training data | `experiments/pilot_random1_player_round_slim.csv`: pilots with a human manager and with a random-parameter rule manager, 8 or 16 rounds | `experiments/2group_8agent_50ep.csv` |
| Manager to beat | DQN, `artifacts/manager/17_exp2_group_payoff_heavy_optimize/model/_manager.pt` (deployed, "pgh", γ 0.98) | `ah` (linear clone of the human managers) |
| Score | payoff per player, summed over the 24 rounds, mean over players and episodes (`temp/s3_stage/experiment_1/evaluation_sims/key_eval_measures.py`) | the group's common pool against `ah` |
| Recorded numbers | deployed RL 658.6, best RL candidate (`gamma_08_heavy`) 662.6, human-like manager clone 550.5, near-zero punishment about 587–592 | – |

The game rules are the same in both: endowment 20, multiplier 1.6, contributions 0–20,
punishment 0–30. Punishment costs the punished player and also comes out of the pool.
Payoff per round is `20 − c − p + (1.6·Σc − Σp)/n`.

**Two blockers found:**
- **The clones are not in the checkout or on S3.** `artifacts/behavioral_cloning/` was
  deleted in 18755cfb. The S3 README promises `experiment_1/models/contribution_clones/`,
  but the manifest has no such entries. The only copies are the LFS objects in
  `.git/lfs/objects/` (from b69c994e). Both load with current code. Retrains with the same
  inputs exist (`raven_script_21`, `raven_script_22`), but their files differ, and the RL
  manager was trained against the originals.
- **The RL manager no longer loads.** Unpickling fails on `EmptyEncoder`, a class removed in
  51ac0ffc (#112). The human-like manager clone (`23_punishment_autoregressive_v4`) loads,
  but runs only unbatched.

## Decisions

Agreed 2026-11-05.

1. **Clones.** The LFS originals of 21/22 v4, the ones the RL manager was trained and
   evaluated against.
2. **What the agents may read.** Only what existed before experiment 1 ran:
   - the pilot data the clones were trained on;
   - reference sims of the clones under zero punishment and under the human-like manager
     clone.

   The live experiment 1 data (`temp/s3_stage/experiment_1/data/rounds.csv`) stays out:
   half of it is the RL manager's own play. The RL manager's sims and numbers stay out too.
3. **Rule inputs.** `c`, `valid` and `p` over all rounds so far, and `t`, each shaped
   `[games, 4, t+1]`. This covers what the RL manager could reach: this round's
   contributions and validity, last round's punishment, the round number, and the history
   its RNN could carry. `in_group` is dropped, since in one group it is always 1.
4. **Timed-out players.** The game is as in the real experiment and in both environment
   versions:
   - a timed-out player is out of that round's pool: their contribution and any punishment
     are zeroed, the pool is shared among the valid players only, and their payoff for the
     round is 0;
   - a punishment of a timed-out player has no effect.

   The only choice is the value written into the contribution field, which the models see
   as input next round (`timeout_contribution`). It is `"default"`, the clones' default of
   15, as in experiment 1's environment, because the clones and the RL manager were trained
   with it. Experiment 2's sweeps use 0. The agent prompt says that a timed-out player's
   `c` reads 15, with `valid` 0, and that the player is not in the pool.
5. **Budget.** The same as the experiment 2 campaign: 12 fresh lineages, then 6 generations
   on the top 4, 36 versions, 256 Sobol points × 500 episodes. The score is the GP peak at
   2000 episodes, with the best of 6 beside it.
6. **Baselines for evaluation.** One manager per game, so these are not opponents. Each runs
   alone with the clones in the head to head and in step 2's reproduction:
   - the deployed RL manager (17, "pgh");
   - the best RL candidate (`gamma_08_heavy`), if its checkpoint can be identified;
   - zero punishment;
   - the human-like manager clone.

## Steps

Work on a branch off `yannik/handover`, merged back when done.

### 1. Experiment 1 assets

- Restore the 21/22 v4 clones from the LFS store to
  `artifacts/behavioral_cloning/{21_contribution_model_v4,22_contribution_valid_model_v4}/`.
  Record their sha256 in `docs/large_files.tsv`, upload them to S3 under
  `experiment_1/models/contribution_clones/` (fixing the manifest's gap), and get them into
  Raven's `artifacts/` (it is bind-mounted read-only into snapshot jobs).
- Make the RL checkpoints load again: a minimal `EmptyEncoder` shim, registered where
  unpickling looks for it. Check it by loading 17 and the other `artifacts/manager/*exp2*`
  candidates.
- Find the `gamma_08_heavy` checkpoint, or record that it cannot be identified.

### 2. Reproduce experiment 1's numbers (gate)

Add the sim config `configs/simulation/manager_testing/28_EXP1_baselines.yml`:
- 1 group, 4 players, 24 rounds, the 21/22 v4 clones, `timeout_contribution: "default"`;
- managers: the deployed RL manager, zero punishment, and the human-like manager clone;
- 1000 episodes each (the RL and human-like managers run unbatched), run on Raven.

The gate: deployed RL at 658.6 and zero punishment at about 590, each within 2 standard
errors. If either misses, find the environment drift (the timeout input value, the clones'
defaults, other changes since the experiment 1 code) before building anything on top. This step also measures the
unbatched RL manager's run time, which sizes step 9.

### 3. The score as a choice of the sweep

- `sweep.py` and `pool_scores.py` take a `score` from the sim config:
  - `pool_vs_anchor`, today's score and the default;
  - `group_payoff`, experiment 1's: summed payoff per player, mean over players and
    episodes.
- The result keeps `pool`/`pool_se` for experiment 2. It adds `score`/`score_se`, and
  `pipeline.py`, `fit_surrogate.py` (`EDGE_EFFECT` scaled to the score's units),
  `campaign.py` and the observation app read `score`.
- `check_sweep` stops requiring an `ah` anchor when the score has none.

This is also the hook that `docs/tasks/a_new_kind_of_autoresearch.md` needs (a score
averaged over several opponents), so it is designed for both.

### 4. The rule engine for one group of 4

- `rule.py`:
  - the rule inputs come from a named input set: `exp2` (today's) or `exp1` (`c`, `valid`,
    `p`, `t`);
  - `dry_run_inputs` takes the player count and group layout;
  - `validate-rule` gets `--setting`.
- `RuleBasedManager` handles one group; today it loops over groups 0 and 1.
- Tests on both settings, including the `device=` check, which carries over unchanged.

### 5. Policy-finder settings

- `scripts/policy_finder/settings/{exp2,exp1}.yml` hold what is now hardcoded:
  - the sim template, the score and the anchor;
  - the agent's data and reference sims (export paths, `instance_paths`);
  - the input set, the dry-run shape, and the agent definition to run.
- `new_instance.sh` takes `--setting`, which defaults to `exp2`. Every step reads it from
  the instance file.
- A `policy-finder-exp1` agent definition shares "How to work" and "Your notes" with today's.
  It describes experiment 1's game, its inputs (decision 3) and its score.
- `docs/reports/basics_exp1.md` covers the game, the pilot data's columns and the reference
  sims, as `basics.md` does for experiment 2.
- `export_instance.py` gets a leak check for the RL manager's artifacts and sims, and for the
  live experiment 1 data (decision 2).
- The reference sims for the agents: `29_EXP1_reference.yml`, with zero punishment and the
  human-like manager clone.

### 6. Smoke

- One `exp1` instance at 16 points × 50 episodes, then its version 2 at full size, as for
  experiment 2.
- Check:
  - the export's contents and the leak check;
  - the image check on Raven (`--setting exp1`);
  - the sweep, the fit, the final run, and the score's units on the page.

### 7. Campaign

`scripts/policy_finder/campaigns/exp1-<date>.yml`, run as before (decision 5). It runs
alongside nothing else, so the 4 agent slots and 16 Raven jobs are its own.

### 8. Categories

Assign the rules' categories with the summarizer. Experiment 1 has no switches, so the
category list will differ from experiment 2's. New categories are drawn up from the rules.

### 9. Head to head on a common seed

`30_EXP1_pf_vs_rl.yml`, run on Raven:
- managers, each alone with the clones: the top 3 lineage bests at their GP peaks, and the
  baselines of decision 6;
- 2000 episodes each, on a seed no sweep or final run used;
- the clones' payoff as the score, with common good beside it.

### 10. Report and artifact

- The report `docs/reports/policy_finder_exp1_vs_rl.md`:
  - the scores per generation;
  - the head to head;
  - what the rules do against what the RL manager does: punishment by round and by
    contribution shortfall, from the head to head's per-round data;
  - cost.
- The observation page, published as a new artifact:
  - its Overview leads with the comparison against the RL GNN manager;
  - its references are the RL manager, zero punishment and the human-like manager clone.

## Cost (estimate)

- **Agents:** 36 runs, about $45 API-equivalent, on the plan.
- **Raven:** the sweeps are smaller than experiment 2's (one group of 4, so about half the
  simulated players). The unbatched RL manager in steps 2 and 9 may dominate; step 2
  measures it.

## Risks

- **The RL numbers may not reproduce** under today's environment code (step 2).
  If they don't, the comparison uses the RL manager re-simulated under the chosen semantics,
  and the report says so.
- **The clones were trained on 8- and 16-round pilots but play 24 rounds.** That is true for
  the RL manager too, so it affects both sides equally. It is stated as a confound.
- **Scope.** Steps 3–5 generalise code that the experiment 2 campaign depends on. Their tests
  must keep the experiment 2 path byte-for-byte unchanged in its sim configs and scores.

## Progress

- 2026-11-05 00:00: step 1 done. The 21/22 v4 clones and the human-like manager clone
  (23 v4) were restored from the LFS store (sha256 match) to `artifacts/behavioral_cloning/`
  here and in Raven's live checkout. The deployed RL manager loads again (`EmptyEncoder`
  and `GraphNetwork.__setstate__`, ecc88be9). `gamma_08_heavy` was
  `18_exp2_gamma_8_heavy`, which is not here, on Raven or on S3: it is left out.
- 2026-11-05 00:35: step 2 gate passed (`28_EXP1_baselines`, job 31913912, 1000 games each).
  Under experiment 1's own formula: deployed RL 660.7 ± 1.8 (recorded 656–659), zero 587.9
  ± 2.2 (587–592), human-like 546.3 ± 3.5 (550.5). The policy finder's score (average payoff
  of the valid players, summed over the game): RL 659.8, zero 591.1, human-like 546.3.
- Steps 3–5 built (d97201e5). One deviation: sweep.json keeps the field `pool` for the score
  under its `metric` (`group_payoff` here) instead of adding `score`, so the pipeline, the
  fit and the campaign read it unchanged. The observation app labels it by `PF_SETTING`.
  The experiment 2 path regenerates byte for byte.
