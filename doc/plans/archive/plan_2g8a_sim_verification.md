# [DONE] Simulate trained 2g8a managers vs AH opponent (issue #93)

## Goal

Settle PR #92's open question — whether the `p ≈ 0` operating point reached by `03_2g8a_{sum,avg}` is the genuine economic optimum under the 50ep AH stack or a reward-hack of the AH response curve. The only way to know is to run the trained managers in simulation against the same opponent AH and switch predictor they trained against, and inspect per-round trajectories. Deliverable is plots posted as a PR #92 comment before merge.

## Plan

| # | Section | Change | Optional |
|---|---|---|---|
| 1 | Sim config | New `configs/simulation/manager_testing/04_2g8a_trained_vs_ah.yml` mirroring the 50ep AH stack, with a new `pairings:` block | |
| 2 | `pairings:` + dynamic dispatch | Add config-driven manager pairings and rebuild per-agent manager assignment each round in `simulate.py` | |
| 3 | `DummyManager` plumbing | Make `simulate.py` manager construction tolerant of `type: dummy` entries (no `path` field) | |
| 4 | Persist per-round frame | Dump concatenated per-round/per-agent dataframe to parquet so trajectory plots can read it | |
| 5 | Pairing-side plot in `create_plots` | New `comparison_pairing_side.jpg` inside the existing inline plotting pipeline — splits each pairing's group_0/group_1 lines | |
| 6 | Run + post PR comment | Execute on Raven, fetch plots, comment on PR #92 with verdict | |

### 1. Sim config — `configs/simulation/manager_testing/04_2g8a_trained_vs_ah.yml`

- **WHAT**: New YAML config. `artificial_humans` block mirrors `configs/simulation/ah_testing/group_switching_ah_punishment_50ep.yml` (same 50ep `group_switching_contribution_50ep` + `raven_script_22` valid + `switch_pred_opt_50ep` stack). `managers:` declares each checkpoint as a named entry: `trained_avg` / `trained_sum` (`type: rl`, paths under `artifacts/manager/03_2g8a_{avg,sum}/model/`), `ah_opponent` (`type: human`, autoreg 50ep punishment AH), `zero_punishment` (`type: dummy`). Episode settings: `switch_every: 4`, `n_episode_steps: 24`, `n_episodes: 100`, `n_groups: 2`, `n_agents: 8`, `agent_groups: [0,0,0,0,1,1,1,1]`, `n_rounds: 24`.
- **WHAT (pairings block)**: Five pairings — `avg_vs_ah`, `sum_vs_ah`, `ah_vs_ah` (baseline + sanity check vs the standalone AH-testing run), `zero_vs_ah` (asymmetric no-punishment side, tests "is p≈0 optimal"), `dummy_vs_dummy` (no-punishment floor — isolates the AHs' natural contribution behavior with no manager signal at all).
- **WHY**: A single config with all four pairings is the cleanest way to get apples-to-apples plots — same AH stack, same seeds, same episode count.

### 2. `pairings:` config key + dynamic per-round dispatch — `src/aimanager/simulation/simulate.py:150-228`

- **WHAT (schema)**: Top-level `pairings:` list of `{name, group_0, group_1}` dicts referencing keys in `managers:`. Each entry becomes one run named `f"ah {h} managed by {pairing.name}"` so the existing `create_plots` filter at `simulate.py:335` keeps matching.
- **WHERE (run construction, `simulate.py:150-156`)**: Gate the new pairings expansion behind `if "pairings" in config`; keep the existing cartesian product as the fallback so legacy configs (`01_compare.yml`, `02_reward_timing.yml`, `03_parallel_groups_*.yml`) work unchanged.
- **WHERE (dispatch, `simulate.py:215-228`)**: Inside the episode loop, build `groups` *each round* from `state["agent_group"]` mapped through the pairing (`group_map = [pairing.group_0, pairing.group_1]; groups = [group_map[g] for g in current_agent_group]`). Legacy single-manager path keeps its static `groups`.
- **WHY**: Training-time `environment.py:347-355` mutates `agent_group` every `switch_every` rounds and `MultiManager.get_punishments` (`api_manager.py:200`) dispatches off the *current* group. A static per-agent assignment would punish post-switch agents with their original manager — opposite of training semantics, would invalidate the comparison.

### 3. `DummyManager` config plumbing — `src/aimanager/simulation/simulate.py:137`

- **WHAT**: Make the `model_path` injection tolerant of entries without a `path` field: `{**v, "model_path": os.path.join(basedir, v["path"])} if "path" in v else {**v}`.
- **WHY**: `DummyManager` (`api_manager.py:150`) takes `**_` and ignores everything, but the current unconditional `v["path"]` access `KeyError`s before `MultiManager.__init__` ever runs.

### 4. Persist per-round frame — `src/aimanager/simulation/simulate.py:577-608`

- **WHAT**: In `run_cli`, dump the concatenated `df` to `output_dir/per_round.parquet` before calling `create_plots`.
- **WHY**: `simulate.py:572` only writes `aggregates.csv` (per-round means), but the trajectory script (§5) needs per-agent per-round data to compute manager-side decompositions and group-size traces.

### 5. Pairing-side plot inside `create_plots` — `src/aimanager/simulation/simulate.py:353`

- **WHAT**: Add an optional `pairings` parameter to `create_plots` and a new plot (`comparison_pairing_side.jpg`) directly after Plot 1. For rows whose `run` matches a pairing, derive `manager_side = pairing.group_{group_id}` and hue by `f"{pairing_name} / {manager_side}"` — splitting each pairing's two sides into separate lines. Wire through `run_cli` with `pairings=config.get("pairings")`.
- **WHY (vs new script)**: The existing `create_plots` already produces ~80% of the deliverables — per-run trajectories (plot 1), group-size evolution (plot 4), switch counts (plot 5), group-switching heatmap (plot 6). Only the within-pairing split (trained side vs opponent side) is missing. Adding one inline plot is ~50 LOC; a standalone script would duplicate most of `create_plots`.
- **WHY (dynamic group_id)**: Under switching, the initial `agent_groups` value is only true for the first `switch_every` rounds. The `group_id` column from `mem_to_df` (`simulate.py:84`) already reflects the dynamic value — use it directly.

### 6. Execute on Raven + PR #92 comment

- **WHAT**: Sync, run `python -m aimanager simulate configs/simulation/manager_testing/04_2g8a_trained_vs_ah.yml` on Raven, fetch with `scripts/fetch_cluster.sh plots/simulation/04_2g8a_trained_vs_ah`, run the plotting script locally on the fetched parquet.
- **WHAT (PR comment)**: Post the manager-side decomposition + group-size + baseline overlay plots with a 2–3 sentence verdict on the reward-hack question.
- **WHY (verdict criteria)**: Compare trained-side payoff (`trained_avg` / `trained_sum` lines from the `*_vs_ah` pairings) against `dummy_vs_dummy` payoff (the no-punishment floor). Trained-side meaningfully *above* the floor → trained policy is doing useful work. Trained-side ≈ floor → policy converged to "do nothing" and the apparent optimality is just the AHs' natural behavior, reward-hack confirmed. `zero_vs_ah` adds the asymmetric case where punishment exists on one side only.

## Implementation notes

- Sanity check: `ah_vs_ah` pairing should reproduce the metrics of the standalone `group_switching_ah_punishment_50ep.yml` run. If it doesn't, the pairings/dispatch refactor is broken — debug before trusting the trained-vs-ah results.
- `reward_mode` (sum/avg) is a training-time reward-shaping knob; verify by reading `environment.py` whether it affects any recorded sim state. If not, omit from the new config.
- `n_episodes: 100` may yield wide error bands under switching variance — bump to 200–500 if first run looks noisy.
- Branch off `69-two-manager-rl-training` (current branch, holds the trained checkpoints). Rebase onto `main` after #92 merges.

## Open questions

- Wire `ah_vs_ah` parity check (vs `group_switching_ah_punishment_50ep.yml`) as an explicit assertion in the script, or just an eyeball check on the plots?

## Next Actions

- [x] Resolve open questions with the user, flip status to `[ACTIVE]`.
- [x] Add `04_2g8a_trained_vs_ah.yml` (§1) — commit `d6d24ff`.
- [x] Edit `simulate.py` — pairings parsing, dynamic dispatch, dummy plumbing, parquet dump (§2–4) — commits `9feceef`, `8bcb76b`.
- [x] Add pairing-side plot inside `create_plots` (§5, revised from standalone script) — commit `9b3c42b`.
- [x] Run on Raven, fetch artefacts. (`ah_vs_ah` parity left as eyeball check; plots line up with the standalone AH-testing run.)
- [x] Post plots + verdict as a PR #92 comment — verdict shifted: dummy_vs_dummy matches trained-side payoff under the new 50ep AH stack, but old-stack control (`05_pr7_02_vs_dummy`) shows dummy clearly worse than RL managers. The "p≈0 exploit" is a 50ep-AH-stack property, not a training failure.
- [x] Move plan to `doc/plans/archive/` and flip to `[DONE]`.

## Outcome

- `comparison_pairing_side.jpg` in `plots/simulation/04_2g8a_trained_vs_ah/` is the headline figure: under the 50ep stack, `dummy_vs_dummy` payoff ≈ trained-side payoff ≈ 25, and the trained policies hold punishment at ~0.
- Old-stack control in `plots/simulation/05_pr7_02_vs_dummy/` shows the opposite pattern (RL ~28–30, dummy ~24, RL managers punish early to bootstrap cooperation, then taper).
- Conclusion: the trained 50ep RL managers correctly identified the optimum *for the AH stack they trained against*. The AH stack itself doesn't model human punishment-responsiveness — data analysis confirmed this is rooted in the dataset's signal structure (ICC 0.64 vs legacy 0.55, marginal ΔR² from punishment 5.6× smaller, ATT roughly half at every contribution level).
- Follow-up direction: AH contribution model audit findings recorded in this conversation; concrete ablations proposed (one-hot prev_punishment encoding, smaller hidden_size, add `agent_group` / `prev_contribution_valid` features). Not opened as an issue yet.
