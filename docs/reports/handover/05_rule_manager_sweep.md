# Rule-based manager search: schema, sweep sim, surrogate, validation

Handover sub-report 5. Scope: the numerical and simulation core of the policy
finder. Launching instances, the agent sandbox and the dashboard are in
sub-report 4. Refs are to `yannik/handover` (bb414f5a). Every file cited here
matches `origin/policy-finder-base` except `src/aimanager/cli.py`, which
adds `unwrap_orchestrator_config` (cli.py:62-85). A sweep config passes
through it unchanged. None of these files has uncommitted changes.

## TL;DR

- A rule is a YAML file with `params` (each with `definition` and `type`), `sweep_config` ranges, optional `constraints`, and `code`. The code is a torch snippet that is `exec`'d once per round over a whole batch of episodes. `manager/rule.py` checks the names the code uses at load time. That check is not a security sandbox: `th.os` can be reached.
- A sweep is 256 scrambled Sobol points (seed 0) × 500 episodes. Each point is a `rule_based` manager in group 0 against `ah` in group 1, played batched (`episode_batch_size: 1000`, batch i seeded `seed+i`). The score is definition 4: the mean over episodes of the group's per-round pool `1.6·Σc − Σp`, with empty rounds counted as 0.
- `±1.1` is the plain standard error over episodes of one point's per-episode pool, with 2000 episodes (sweep.py:66). It does not account for choosing the best of the ~6 validated points.
- `ah` is **not a GNN**. It is a multinomial logistic regression punisher from sklearn, with a severity copula (ρ = 0.4273): `artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib`. The sweeps run on Levin's stack (24_LEVIN AH block). `zero` is `DummyManager(0)`.
- The surrogate is a local sklearn GP: Constant × Matern-5/2 ARD + White, with per-point `alpha = se²`. Its peak is a candidate only. The "final best" is the top of a fresh-seed rerun (seed 1000, 2000 episodes) of the peak plus the 5 best measured points.
- There are no rule-based managers on this branch (`configs/managers/rule_based/` holds only `.gitkeep`). Decay and sigmoid exist only on `origin/autoresearch-optimized-stack`, in an **old schema that this branch's `validate-rule` rejects**.
- Main Raven risks:
  - the image pins `scikit-learn==1.5.2`, but the `ah` joblib was pickled with 1.6.1
  - `/sync_to_cluster` excludes `artifacts/`, so the stack must already be in the shared checkout
  - rules are validated locally on torch 2.5 CPU but run on torch 1.11 CUDA
  - `run_sweep.sh`/`run_validate.sh` are ssh/rsync based

## Command chain

```
LOCAL (proxy container, CPU, py3.10/torch 2.5.1)          RAVEN (algoinst.sif, A100, py3.9/torch 1.11)
configs/managers/rule_based/<rule>.yml   (must be inside the checkout: gen_sim_config.py:116-119)
 │ python -m aimanager validate-rule <rule.yml> [--max-params N|none]
 │ python scripts/policy_finder/generate_sim_config.py --config <rule.yml> [--n-parts 2]
 ▼   (run_sweep.sh uses n_parts = ceil(points·500 / 125000) = 2 for 256 points)
configs/simulation/policy_finder/<rule>_sweep[_p<k>of<N>].yml
 │ POST /sync_to_cluster  ───────────────────────────────▶ /u/ykeller/algorithmic-institutions
 │ POST /submit_slurm_job per part ──────────────────────▶ python -m aimanager simulate <part.yml>
 │                                                         → plots/simulation/policy_finder/<part>/
 │                                                           sweep.json, per_round.parquet, 2 .jpg
 │ POST /run_in_container: cp <part dirs> /opt/cluster_transfer/… ◀─┘
 │ POST /sync_from_cluster  → /var/cluster_transfer/…  → copy into plots/simulation/policy_finder/
 ▼
python scripts/policy_finder/merge_sweeps.py <part dirs> --out plots/…/<rule>_sweep/sweep.json   (parts only)
python scripts/policy_finder/fit_surrogate.py plots/simulation/policy_finder/<rule>_sweep   → landscapes.json
python scripts/policy_finder/generate_sim_config.py --validate plots/…/<rule>_sweep/sweep.json [--top 5]
 ▼   → configs/simulation/policy_finder/<rule>_sweep_validate.yml (seed 1000, 2000 ep/point)
 │ sync, submit ─────────────────────────────────────────▶ simulate → plots/…/<rule>_sweep/validate/
 │ fetch as above ◀──────────────────────────────────────┘
 ▼
plots/…/<rule>_sweep/validate/sweep.json  `best` = the rule's final params (loads as a manager's `params:`)
```

On the old path, `run_sweep.sh` and `run_validate.sh` ran these steps over ssh/rsync. The Raven side of every step above is `python -m aimanager simulate`.

## 1. Rule schema and loading (`src/aimanager/manager/rule.py`)

- **Schema** (docstring rule.py:6-29). The allowed top-level keys are `params`, `sweep_config`, `constraints` and `code` (`RULE_KEYS`, :59).
  - Each param is exactly `{definition: str, type: int|float}` (:89-103).
  - `sweep_config` entries take one of three forms (:106-140):
    - a fixed number
    - `[low, high]`
    - `[low, high, log]`, which needs `low > 0`
  - An `int` range needs integer bounds.
  - Param names must not clash with the inputs, `th`, `punishment` or `best` (:171-175).
- **What the code sees.** `RULE_INPUTS` (:43-55) are float tensors `[B, A, 1]`, one value per player at punish time: after round t's contributions and before its punishments.
  - The player: `c, valid, c_prev, p_prev, t`. In round 0, `c_prev = c` and `p_prev = 0`.
  - Own group: `n, c_group`. `c_group` is the mean of the valid others.
  - Other group: `n_other, c_other, p_prev_other`.
  - Means are taken over valid players, and a mean over nobody is 0 (`rule_inputs`, :274-319).
  - Params arrive as 0-d float tensors on the input device, so `int` params are floats too (:334).
- **Load-time checks** (`read_rule`, :143-214):
  - The AST name check: the code may read only inputs, `th`, declared params and names it assigns itself. It must read every declared param and assign `punishment`. Imports are rejected (:70-80).
  - Constraints must be `Compare` expressions over params only.
  - `check_params` (:217-246): the param set must match the declaration exactly, every value must be a finite number, `int` params must be integers, and every constraint must hold (via `eval` with empty builtins).
  - `load_rule` (:249-266) takes a JSON path or an inline mapping. If the mapping has a `best` key, it uses `best`, which is why a `sweep.json` loads directly as params.
- **Execution** (`run_rule`, :322-345):
  - `exec(code, {"__builtins__": {}, "th": th, **params, **cloned inputs})`.
  - A scalar result is broadcast to every player. A NaN raises `ValueError`, which kills the whole sim.
  - `RuleBasedManager._punish` (api_manager.py:203-205) clamps to `[0, n_punishments-1] = [0, 30]` and casts to int64.
  - Batched path: `batched_punish` (:207-218) builds the inputs from env state. Per-episode path: `get_punishments(rounds)` (:220-240).
  - **This is not a sandbox.** `_names` inspects only `ast.Name` nodes, so attribute chains are unchecked. Locally, `_names(parse("x = th.os.getcwd()…"))` returns `{'c','k','th'}`, and `hasattr(torch, 'os')` is True. Agent-written code runs on Raven as `ykeller`.
- **Sobol design** (`sobol_design`, :371-420):
  - `scipy.stats.qmc.Sobol(d=#ranged, scramble=True, seed=0)`, drawn with `random_base2`. `n_points` must be a power of two.
  - Log ranges are mapped through `exp`.
  - An `int` range is drawn over `[low-0.5, high+0.5]`, rounded and clipped. Points that become identical are merged, so a design can hold fewer than 256 points.
  - scipy is imported lazily (:383), so the Raven sim never needs it.
- **Validation** (`validate_rule`, :423-453):
  - `sweep_config` is required, and the number of declared params must be in `[min_params, max_params]`. The default is 1–4; pass `none` for no upper bound (#244).
  - Every design point must pass `check_params` and then run without error or NaN on 2048 random CPU states (`dry_run_inputs`, :348-362).
  - CLI: `python -m aimanager validate-rule <rule.yml> [--min-params N] [--max-params N|none] [--sobol-points N]` (cli.py:156-170, 192-206). It prints `Valid: …` or `Invalid: …` and exits 1 on failure.
- **Rules on this branch:** `configs/managers/rule_based/` contains only `.gitkeep`, the same as `policy-finder-base` (#235).
  - `origin/autoresearch-optimized-stack` has `decay.yml`, `sigmoid.yml` and `params/{decay_k1,k2,k4,k8, sigmoid_opt_pool, sigmoid_best_cap10_pool}.json`.
  - Those files use the **old schema**: params are plain strings, there is no `sweep_config`, and the code reads only `c, t`.
  - No instance branches (`policy-finder/<name>`) exist on `origin` or locally, and there is no `../policy-finder-worktrees`. The agent-written rules behind the result page (e.g. pf-20261008-3 v6) are **not in this clone**.

## 2. Sweep sim

- **`scripts/policy_finder/generate_sim_config.py`.**
  - `TEMPLATE` is `configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml` (:53-55). The script copies its stack, its sim parameters and `episode_batch_size: 1000`. It also copies `save_per_round: true`, so a sweep writes `per_round.parquet`.
  - It adds:
    - one manager per point, `<rule>_s<NNN>`, with its params inline
    - the `ah` manager from the template
    - one pairing per point, `<rule>_s<NNN>_vs_ah`
    - `seed` (default 42), `n_episodes` (default 500), `output_dir`, `figure_name`, and `sweep: true` (:93-150)
  - It refuses a rule name containing `_vs_` (:112), and a rule outside `ROOT` (:116-119).
  - `--n-parts N` splits the design into contiguous parts. Part k's seed is offset by the batches of the earlier parts, so no batch seed repeats (:153-175). In a local check, parts of 4×500 got seeds 42 and 44.
  - `--validate <sweep.json> [--top 5] [--n-episodes 2000] [--seed 1000]` (:178-227) writes `<rule>_sweep_validate.yml`.
    - The points are the GP peak, named `<rule>_gp`, plus the top-k measured points under their sweep names. A measured point identical to the peak is dropped.
    - Output goes to `<rule>_sweep/validate/`. The landscapes file is required.
  - Other options: `--out` (single part only) and `--root`, which sets the repo that paths are relative to.
- **The `sweep: true` path.**
  - `run_cli` first runs `check_sweep` (simulate.py:814-816, sweep.py:29-54). It requires:
    - an `ah` manager
    - every pairing with a point in `group_0` and `ah` in `group_1`
    - every point a `rule_based` manager with inline params
    - a single rule shared by all points
  - After the sim, `write_sweep` writes `sweep.json` and `plot_best` draws `comparison_pairing_side.jpg` and `group_size_evolution_global.jpg` for the best point only. All other plots are skipped (simulate.py:850-854).
  - `simulation/run.py` is the sbatch orchestrator for the manual path only. Do not submit it.
- **Batching** (#232; simulate.py:140-155, 209-298, 401-412).
  - `episode_batch_size` must be in 1..20000.
  - Episodes of all runs that share an AH block are chunked in config order, so with 500 episodes per point a batch of 1000 holds 2 points.
  - In each round, every manager used in the batch punishes the **whole** batch, and `th.where` picks each agent's manager from its current `agent_group` (:279-289).
  - Managers without `batched_punish` fail early. `rl` and `human` both lack it.
  - Records stay on the device until `batches_to_dfs` (:301-327).
- **Seeding.** `run_cli` seeds `random`, `numpy`, `torch` and CUDA with `seed` (:823-830). `run_batched` then reseeds before every batch with `seed + batch_index` (:257-264). Consequences:
  - results depend on `episode_batch_size`
  - the RNG streams are not shared between design points, so there are no common random numbers across points
  - every rule's sweep reuses seeds 42..169
  - the validation seed of 1000 is past these, as long as a sweep has fewer than 958 batches
- **Definition 4** (`simulation/pool_scores.py`, shared with `scripts/plotting/plot_winrates.py`):
  - `add_pool` gives each agent `1.6·c − p`, where `p` is the punishment as charged; `env.punish` zeroes it for timed-out players (:18-21).
  - `group_round_scores` sums per (episode, round, group). An empty group-round counts as 0 (:28-35).
  - The episode mean is taken over its 24 rounds. `against_anchor` collects each manager's group in every pairing whose *other* side is the anchor; the anchor's own score comes from both groups of `ah_vs_ah` (:48-69).
  - `sweep_result` gives each point `pool` (mean over episodes), `pool_se = std(ddof=1)/sqrt(n)`, `members` (mean group size) and `n_episodes` (sweep.py:57-92). `best` is the plain argmax of `pool`.
- **Where ±1.1 comes from.** It is that point's `pool_se` in `validate/sweep.json`, over 2000 episodes; the per-episode sd is about 49, which follows from 1.1·√2000.

## 3. Surrogate (`scripts/policy_finder/fit_surrogate.py`) and merge

- **Axes.** Each ranged param is an axis, mapped to [0, 1] (in log for `log` ranges). Fixed params are not axes (:83-124).
- **Kernel.**
  - `ConstantKernel(1, (1e-3, 1e4)) * Matern(ls=0.5 per axis, bounds (0.01, 100), nu=2.5) + WhiteKernel(0.1, (1e-6, 1e2))`
  - `GaussianProcessRegressor(alpha=(se/std(y))², normalize_y=True, n_restarts_optimizer=8, random_state=0)` (:132-157)
  - ConvergenceWarnings are silenced on purpose.
- **Peak** (:160-193). It scores 2^14 Sobol candidates, with int axes snapped to integers. It then polishes the top 16 with L-BFGS-B along the float axes and keeps the best one that passes `check_params`, constraints included.
- **Profiles** (:196-218) are cuts through the peak along each axis: 101 points per float axis, every integer for an int axis. `effect` is the max minus the min of the profile mean. `at_edge` lists params whose peak lies within 0.02 of a range end and whose effect is at least 1.0 (:79-80, 281-285).
- Run as `fit_surrogate.py <sweep dir> [--rule <yml>] [--seed 0]`. It writes `<dir>/landscapes.json` and prints a summary. It runs locally in about 10 s for 256 points.
- The docstring cites `scripts/rule_sigmoid/fit_surrogate.py` (:8), which is not in this branch. It exists only in history: `git log --all` finds it in 990afd4e.
- **`merge_sweeps.py`** (:24-57, sweep.py:140-170) joins part dirs named `<sweep>_p<k>of<N>`. It checks that all parts 1..N are present and that they agree on rule, anchor, score, n_episodes and episode_batch_size. It sorts points by index, rejects duplicates, recomputes `best`, replaces `seed` with a `seeds` list, and copies the best part's two plots.

## 4. Opponent, stack and reference numbers

- **`ah`** (template :16-19) is `type: linear`, served by `LinearManager` (api_manager.py:243-274) and, batched, by `LinearPunisherOpponent` (manager/linear_opponent.py).
  - The bundle, inspected read-only: sklearn `LogisticRegression`, 31 levels, with a `StandardScaler` over 7 local features: `contribution, contribution_max, contribution_valid, prev_contribution, prev_punishment, round_number, is_first`.
  - Plus a group-round severity copula `rho=0.4273` and `sample=True`.
  - Its pickle records `sklearn_version 1.6.1`.
  - Inference runs on CPU: numpy `predict_proba` and `th.special.ndtr` (:155). Only the results go back to the device.
- **The stack** is the 24_LEVIN AH block: contribution `group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/…epochs_575.pt`, valid `raven_script_22/…dataset_full.pt`, switch `switch_joint_exodus/…50ep_doubled.pt`.
  - It is identical to `24_LEVIN_vnode_skip_timeoutpun_self_gnncopar1_contr_gnn_switch.yml` (checked with a diff), and is also the stack of `25_LEVIN_run3_sigmoid_pairings*` on the optimized-stack branch.
  - All four artifacts are real files in the working tree (zip archive version 3, not LFS pointers), and git-tracked.
- **`zero`** is `DummyManager(constant_punishment=0)` (api_manager.py:170-187).
- **Reference numbers.**
  - **56.9 / 69.8 / 59.5 are not in the repo.** The result page says they are reruns at validation conditions: 2000 episodes, seed 1000, the #226 run-3 stack.
    - They are stored in `references.json` under `/u/certuer/policy-finder-observatory/` on Raven, another user's home.
    - No config for the rerun is on this branch. `opt` = `sigmoid.yml` + `params/sigmoid_opt_pool.json`, only on `origin/autoresearch-optimized-stack`.
  - The 69.8 inside `docs/reports/zero_punisher_test.html` is a different number: zero's pool in its head-to-head against k1.
  - That report's definition-4 values (1000 episodes, seed 42) are AH 56.0, zero 60.5, opt 71.0.
  - I recomputed them read-only from `plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet` using `pool_scores`: **ah 55.99 ± 0.99 (n=2000), zero 60.46 ± 0.95 (n=2000), members 4.00 / 4.57**. Both match the report.

## 5. File formats

- **Sweep sim config.** The template's keys, plus:
  - `managers: {ah: …, <rule>_s000: {type: rule_based, rule: configs/managers/rule_based/<rule>.yml, params: {…}}, …}`
  - `pairings: [{name, group_0, group_1: ah}]`
  - `sweep: true`
  - `output_dir: plots/simulation/policy_finder/<rule>_sweep[_pkofN]`
- **`sweep.json`** (sweep.py:78-92):
  - `best` (params dict) and `best_name`
  - `rule` (relative path as given in the config) and `anchor: "ah"`
  - `score` (a text description) and `seed`
  - `n_episodes` and `episode_batch_size`
  - `points: [{name, params, pool, pool_se, members, n_episodes}]`
  - A merged file swaps `seed` for `seeds: [...]`. `validate/sweep.json` has the same format, with points named `<rule>_gp` and `<rule>_sNNN`.
- **`landscapes.json`**:
  - `rule`
  - `fit: {kernel (str), n_points, median_se, extra_noise_sd, fit_rmse}`
  - `params: {<axis>: {range, log, type, effect, length_scale}}`
  - `peak: {params (all params, fixed included), mean, sd, at_edge: [...]}`
  - `points: [{name, mean, sd}]`, in sweep order
  - `profiles: {<axis>: {x, mean, sd}}`
- **Run outputs:** `per_round.parquet` has the columns of `mem_to_df` (simulate.py:55-103). Each part numbers its episodes from 0.

## 6. Local test results (this container: Python 3.10.12, torch 2.5.1+cpu, sklearn 1.6.1, scipy 1.14.1, numpy 1.26.4, PyG 2.6.1)

- `PYTHONPATH=src python -m pytest src/aimanager/tests/test_rule_validate.py src/aimanager/tests/test_sweep.py -q`: **35 passed, 11 warnings in 8.58s**. The warnings are matplotlib pyparsing deprecations.
- `validate-rule` on the optimized-stack rules, extracted with `git show` to the scratchpad:
  - `sigmoid.yml` fails with `Invalid: …/sigmoid.yml: param `p_max` must declare exactly ['definition', 'type']` (exit 1).
  - `decay.yml` fails with `Invalid: …/decay.yml: param `k` must declare exactly ['definition', 'type']` (exit 1).
- `validate-rule` on scout rules written to the scratchpad:
  - A 2-param `scout_shortfall.yml` (shown below) passes: `Valid: …, 256 distinct design points of 256 Sobol points`.
  - A port of sigmoid to the new schema fails at the default limit with `Invalid: …: declares 5 params, allowed 1 to 4`, and passes with `--max-params 5` (256 distinct points).
  - `--sobol-points 6` fails with `Invalid: sobol points must be a power of two, got 6`.
- `generate_sim_config.py` with `--root <scratch tree> --out <scratch>`:
  - Tiny 4×64 config: `4 design points x 64 episodes = 256 episodes`.
  - `--n-parts 2 --sobol-points 8` produced two parts with seeds 42 and 44.
  - `--validate` on synthetic sweeps produced `6 points x 2000 episodes = 12000 episodes` at seed 1000, output `…/<rule>_sweep/validate`.
- `fit_surrogate.py` on a **synthetic** `sweep.json` (made-up pools, se 2.2, true peak 72 at k=1.2, norm=11):
  - 256 points, 10 s: `peak {'k': 1.147, 'norm': 11}: 70.5 +- 0.6`, and `measured best s062: 74.1 +- 2.2, surface 69.8`. The measured best reads high and the GP shrinks it, which is the #238 effect.
  - 4 points: the fit is degenerate but the script runs.
- The repo working tree was not modified by any of this: `git status` still lists 282 entries of its own, plus sibling sub-report 04.

## 7. Portability to the Raven image (`algoinst.sif`, run as `python -m aimanager simulate`)

- **Python 3.9 syntax:** OK. `ast.parse(feature_version=(3, 9))` passes for every scoped file, and none of them has runtime PEP 604 annotations.
- **Torch 1.11 APIs:** nothing in the scoped code is newer than 1.11. The calls are `broadcast_to`, `special.ndtr`, `searchsorted`, `where` with tensor arguments, and `gather`. `ndtr` and the linear punisher also ran on Raven before, on the manual `.venv` path with the same torch.
- **Dependencies:** the code needs scipy, which is in pyproject.toml:17, and sklearn, in pyproject.toml:16. The GP and Sobol code is local only.
- **Risk 1: sklearn version skew.** `requirements.raven.txt:10` pins `scikit-learn==1.5.2`, while `pyproject.toml:16` requires `>=1.6.1` and `uv.lock` (py3.9) locks 1.6.1. The `ah` bundle was pickled with sklearn 1.6.1. Unpickling it in 1.5.2 should at least emit `InconsistentVersionWarning`. Whether it loads correctly is **unverified**, because no 1.5.2 is available locally. numpy (<1.24 in the image, 1.26.4 in the lock) and pandas (2.2.3 vs 2.3.3) also differ from `uv.lock`.
- **Risk 2: artifacts are not synced.** `/sync_to_cluster` excludes `artifacts/` and `plots/` (run_on_cluster skill). On the manual path, isolated dirs received the AH artifacts (simulate_cluster.sh:72-80), but the canonical checkout never did. The four stack files and the joblib must already exist under `/u/ykeller/algorithmic-institutions/artifacts/`, and this is unknown. Results also come back only through `/opt/cluster_transfer`.
- **Risk 3: the rule is validated on a different torch and device than it runs on.**
  - `validate-rule` used to run in the instance's `$MAIN/.venv` (torch 1.11; new_instance.sh:109). That `.venv` does not exist in this container, where the system Python has torch 2.5.1 CPU.
  - Agent code that is legal in 2.5 can fail in 1.11 CUDA and then kill a whole part. Examples:
    - `th.where(cond, x, 0.0)`: I expect a dtype error before 1.12, but this is unverified.
    - CPU tensors combined with CUDA inputs, e.g. `th.arange(24)`.
  - Running `validate-rule` inside the image through `/run_in_container` closes the version gap. It does not close the device gap.
- **Risk 4: the scripts assume ssh/rsync.**
  - `run_sweep.sh:53-55, 71-74, 84, 97` and `run_validate.sh:51-54, 61-62, 69, 75, 81` use `raven_remote_dir.sh`, `~/ai-isolated/<branch>` (from `.raven_remote_dir`, present on this branch), `simulate_cluster.sh`, `squeue` and `rsync`.
  - They also run from worktrees in `../policy-finder-worktrees`, which `/sync_to_cluster` never ships, and they `git commit` the results.
  - The Python scripts themselves are path-agnostic: `generate_sim_config.py` takes `--root`, and `fit_surrogate.py`/`merge_sweeps.py` take directories.
  - The rule and the sweep configs must be in `/opt/algorithmic-institutions` before syncing. A sync from a tree that lacks another instance's configs may delete them remotely: rsync's delete behaviour for non-excluded paths was not checked.
- **Resources:** two parts of 64k episodes plus one 12k-episode validation, each within the 16 GB / 1 h simulation defaults (run_sweep.sh:29; docstring gen_sim_config.py:10-13). That leaves room for one sweep at a time under the 4-job cap.
- **Proposed cheapest Raven test.** First, a CPU preflight via `/run_in_container`, as one line:
  `cd /opt/algorithmic-institutions && ls -la artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib artifacts/artificial_humans/{group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula,raven_script_22,switch_joint_exodus}/model/ && python -m aimanager validate-rule configs/managers/rule_based/scout_shortfall.yml; python -W always -c "import joblib; b=joblib.load('artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib'); print(b['estimator'].predict_proba(b['scaler'].transform([[5,0,1,9,0,3,0]])).shape)"; python -c "import torch; c=torch.arange(4.); print(torch.where(c>1, c, 0.0))"`
- **The GPU job.** Add the two files below, sync, then submit:
  - body: `{"cluster":"raven","program_call":"-m aimanager simulate configs/simulation/policy_finder/scout_shortfall_sweep.yml","time":"00:20:00","n_gpu":1,"n_nodes":1,"n_cpu":2,"launcher":"python","run_group":"scout-sweep","project":"algorithmic-institutions"}`
  - Expected in the log: `Using device: cuda`, `Batched: 4 runs x 64 episodes in 2 batches of up to 128`, `Start batch 0/1: 128 episodes`, then `Saved: …/sweep.json`.
  - Fetch the result, run `fit_surrogate.py` on it locally, and optionally run `--validate --top 2 --n-episodes 64` as a second tiny job.
  - Remove both files afterwards, so that `policy-finder-base` stays clean (#235).

`configs/managers/rule_based/scout_shortfall.yml`:

```yaml
# Handover scout rule (not an agent's rule): a fine per point below a fixed
# norm, timed-out players spared. Two params, one int, one log-scaled float.
params:
  k:
    definition: punishment per point of contribution below the norm
    type: float
  norm:
    definition: contribution at or above which nobody is punished
    type: int
sweep_config:
  k: [0.25, 4, log]
  norm: [0, 20]
constraints:
  - norm >= 0
code: |
  punishment = k * th.clamp(norm - c, min=0) * valid
```

`configs/simulation/policy_finder/scout_shortfall_sweep.yml` is the output of `generate_sim_config.py --config configs/managers/rule_based/scout_shortfall.yml --sobol-points 4 --n-episodes 64`, with one hand edit: `episode_batch_size: 1000 → 128`, so that two batches exercise the `seed + i` reseeding and the mixed pairings:

```yaml
seed: 42
artificial_humans:
  group_switching:
    contribution_model: artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
    valid_model: artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt
    switch_model: artifacts/artificial_humans/switch_joint_exodus/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt
managers:
  ah:
    type: linear
    path: artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib
  scout_shortfall_s000:
    type: rule_based
    rule: configs/managers/rule_based/scout_shortfall.yml
    params: {k: 2.643303108341821, norm: 19}
  scout_shortfall_s001:
    type: rule_based
    rule: configs/managers/rule_based/scout_shortfall.yml
    params: {k: 0.8743360697483398, norm: 3}
  scout_shortfall_s002:
    type: rule_based
    rule: configs/managers/rule_based/scout_shortfall.yml
    params: {k: 0.4982507569341006, norm: 12}
  scout_shortfall_s003:
    type: rule_based
    rule: configs/managers/rule_based/scout_shortfall.yml
    params: {k: 1.2627890232909804, norm: 6}
pairings:
- {name: scout_shortfall_s000_vs_ah, group_0: scout_shortfall_s000, group_1: ah}
- {name: scout_shortfall_s001_vs_ah, group_0: scout_shortfall_s001, group_1: ah}
- {name: scout_shortfall_s002_vs_ah, group_0: scout_shortfall_s002, group_1: ah}
- {name: scout_shortfall_s003_vs_ah, group_0: scout_shortfall_s003, group_1: ah}
switch_every: 4
n_episode_steps: 24
n_episodes: 64
n_groups: 2
n_agents: 8
agent_groups: [0, 0, 0, 0, 1, 1, 1, 1]
n_contributions: 21
n_punishments: 31
n_rounds: 24
timeout_contribution: 0
episode_batch_size: 128
output_dir: plots/simulation/policy_finder/scout_shortfall_sweep
figure_name: scout_shortfall_sweep
save_per_round: true
basedir: .
sweep: true
```

## Open questions

- Are the 24_LEVIN `.pt` files and the `ah` joblib present in `/u/ykeller/algorithmic-institutions/artifacts/`? If not, they need one manual copy, because `/sync_to_cluster` will not ship them.
- Does the image's sklearn 1.5.2 unpickle and score the 1.6.1 bundle identically? If not, should the image pin `scikit-learn==1.6.1`, matching pyproject and uv.lock?
- Does the slurm template run `python -m aimanager simulate` with cwd set to the checkout root? Every config path is relative (`basedir: .`).
- Where are the configs behind 56.9 / 69.8 / 59.5, and can `ykeller` read `/u/certuer/policy-finder-observatory/`? To rerun `opt` on this branch, `sigmoid.yml` must first be ported to the new schema.
- Where are the instance branches holding the 13 lineages and 32 versions, such as `policy-finder/pf-20261008-3`, and their `sweep.json`/`landscapes.json` files? They are not in this clone.
- `±1.1` is the standard error of one point. The final best is the maximum of about 6 validated points, which biases it slightly upward: if all 6 were equal, the bias would be about +1.3 se. Should a third fresh seed confirm the winner?
- Does `/sync_to_cluster` delete remote configs that are missing locally, and does that matter for parallel instances sharing the one checkout?
- Should validation of rule code run inside the image, on torch 1.11, by default? It could also run on CUDA (on the `claude-gpu` service, or via `/run_in_job`) to catch device mismatches.
