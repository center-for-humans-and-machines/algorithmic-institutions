# [ACTIVE] Counterfactual contribution diagnosis under sustained punishment

Tracks GitHub issue #94. Branch: `94-counterfactual-contribution-diagnosis`.

## Goal

PR #92 + PR #95 established that RL managers trained against the **50ep AH
stack** converge to `punishment ≈ 0` and tie a do-nothing dummy on payoff —
while the same dummy lags RL by 5–6 payoff under the **old BC stack**. PR #95's
verdict: the 50ep `group_switching_contribution_50ep` model is itself
exploitable — weakly elastic to punishment, autoregressive on
`prev_contribution`.

Issue #94 asks the diagnostic counterfactual: **under a sustained constant
punishment held across all rounds, does the 50ep contribution model's
autoregressive trajectory decay (physiologically sensible) or stay flat (the
exploit signature)?** We answer this by parametrizing `DummyManager` so its
constant level is configurable, then sweeping `p ∈ {0, 5, 10, 20, 30}` through
the existing simulation pipeline. Per-round trajectories fall out for free;
no probe infra, no new plotting.

One PR covers both contribution models — the old-vs-new comparison is just a
second config pointing at different AH artifacts; no extra code, no reason to
split.

## Plan

| # | Section                        | Change                                                                                                          | Optional |
|---|--------------------------------|-----------------------------------------------------------------------------------------------------------------|----------|
| 1 | Parametrize `DummyManager`     | Add `constant_punishment: int = 0` arg; return `full_like(...)` tensor.                                         |          |
| 2 | 50ep sweep config              | New `configs/simulation/manager_testing/06_sustained_p_sweep_50ep.yml` with one dummy per `p ∈ {0,5,10,20,30}`. |          |
| 3 | Old-BC sweep config            | New `configs/simulation/manager_testing/07_sustained_p_sweep_v4bc.yml`, same dummy ladder, v4 BC AH stack.      |          |
| 4 | Run both sweeps on Raven       | `python -m aimanager simulate` via existing cluster scripts; fetch artifacts.                                   |          |
| 5 | Old-vs-new overlay             | One-off notebook reading both `per_round.parquet` files, overlay trajectories per `p`-level.                    |          |
| 6 | Verdict write-up               | Brief issue comment on #94 with overlay plot + per-round parquet paths.                                         |          |

### 1. Parametrize `DummyManager`

- File: `src/aimanager/manager/api_manager.py:150`.
- Add `constant_punishment` kwarg (default `0`, cast to `int`) to `__init__`;
  store on `self`.
- Change `get_punishments` to return
  `th.full_like(data["punishment"], self.constant_punishment)` instead of
  echoing the zero-default tensor from `data["punishment"]`.
- WHY: the current dummy only happens to output zero because the default
  `punishment` field in the env-state dict is zero; we need an explicit,
  per-instance constant level so multiple dummies at different `p` levels can
  coexist in a single sim config.
- Backward compatibility: default `constant_punishment=0` keeps every existing
  config (`04_2g8a_trained_vs_ah.yml`, `05_pr7_02_vs_dummy.yml`, etc.)
  bit-identical.
- The `**_` swallow remains so unrelated kwargs (`n_steps`, `path`) injected by
  `MultiManager` (`src/aimanager/manager/api_manager.py:172`) still pass
  through.

### 2. 50ep sweep config

- File: `configs/simulation/manager_testing/06_sustained_p_sweep_50ep.yml`
  (mirror layout of `04_2g8a_trained_vs_ah.yml`).
- `artificial_humans.group_switching` block: identical to config 04 (50ep
  contribution / valid / switch models).
- `managers` block: one `dummy_p{k}` entry per `k ∈ {0, 5, 10, 20, 30}`, each
  `type: dummy` with `constant_punishment: k`.
- `pairings` block: 6 entries — symmetric ladder `p0/p0`, `p5/p5`, `p10/p10`,
  `p20/p20`, `p30/p30` (clean per-level baseline) plus `p0/p30` (extreme
  contrast — both trajectories land in one panel via the existing
  `comparison_pairing_side.jpg` split, and lets us see whether agents migrate
  toward the unpunished side under switching).
- Reuse the rest verbatim from config 04: `switch_every: 4`,
  `n_episode_steps: 24`, `n_episodes: 100`, `n_groups: 2`, `n_agents: 8`,
  `agent_groups: [0,0,0,0,1,1,1,1]`, `n_contributions: 21`,
  `n_punishments: 31`, `n_rounds: 24`.
- `output_dir: plots/simulation/06_sustained_p_sweep_50ep`;
  `figure_name: sustained_p_sweep_50ep`.

### 3. Old-BC sweep config

- File: `configs/simulation/manager_testing/07_sustained_p_sweep_v4bc.yml`,
  same dummy ladder as config 06.
- `artificial_humans` block points at the v4 BC stack
  (`21_contribution_model_v4` / `22_contribution_valid_model_v4`); see
  `configs/simulation/manager_testing/05_pr7_02_vs_dummy.yml` for the exact
  paths and any compat flags. **Verify these checkpoints still load against
  the current `GraphNetwork.load` interface before running the full sweep**
  — flag in cluster smoke test step.
- Episode settings: match the old-stack reference (1g4a in config 05) — i.e.
  `n_groups: 1`, `n_agents: 4`, `agent_groups: [0,0,0,0]`. The old BC model
  was trained on 1g4a data; running it under 2g8a would be off-distribution.
- `pairings`: 5 entries, one dummy per `p`-level managing the single group.
  No cross-level extreme pairing (only one group exists in 1g4a).
- `output_dir: plots/simulation/07_sustained_p_sweep_v4bc`;
  `figure_name: sustained_p_sweep_v4bc`.
- Caveat for the write-up: 50ep is 2g8a, v4 BC is 1g4a — group-mean
  trajectories are still comparable per-agent, but absolute payoffs aren't.
  Frame the comparison around contribution decay / elasticity shape, not
  level.

### 4. Run both sweeps on Raven

- Submit via the existing simulation cluster pathway used in PR #95 (see
  `scripts/simulate_cluster.sh` / `src/aimanager/simulation/run.py`).
- Fetch results with `scripts/fetch_cluster.sh
  plots/simulation/06_sustained_p_sweep_50ep` and the same for `07_*`.
- Expected artifacts per sweep (produced unchanged by `simulate.py`):
  `per_round.parquet` + `comparison_pairing_side.jpg` + group-size /
  switch-count plots.
- Smoke step before the full run: 5 episodes per sweep — verifies the v4 BC
  checkpoints load and that both configs run end-to-end.

### 5. Old-vs-new overlay

- One-off plotting notebook or short script (under `notebooks/` or
  `scripts/plotting/`) reading both `per_round.parquet` files.
- Output: one figure with 5 panels (one per `p`-level), each panel overlaying
  the 50ep and v4 BC mean-contribution trajectory. No pipeline change.

### 6. Verdict on the issue

- Post a short comment on issue #94 with:
  - The overlay figure + per-round parquet paths.
  - One-line read of the trajectories: does 50ep contribution at `p=0` stay
    flat while v4 BC's decays? Same check at `p=30` (do both saturate?).
  - Verdict on the original PR #92 question: model-side flatness vs
    data-coverage gap.
- This PR closes #94.

## Out of scope

- Retraining any AH model.
- Touching the RL pipeline or manager training.
- Modifying the switch predictor or punishment AH.
- Extending `intervention_probe.py` with window-intervention concepts.

## Next Actions

Walk through with the user step-by-step — no engineer-agent handoff.

- [ ] Flip status to `[ACTIVE]`.
- [ ] Edit `DummyManager` in `src/aimanager/manager/api_manager.py`.
- [ ] Write config 06 (50ep).
- [ ] Write config 07 (v4 BC) — smoke-check v4 BC checkpoints load first.
- [ ] Submit both sweeps on Raven; fetch artifacts.
- [ ] Write overlay notebook / script; commit figure + parquets.
- [ ] Open PR referencing #94; post verdict comment with overlay.
