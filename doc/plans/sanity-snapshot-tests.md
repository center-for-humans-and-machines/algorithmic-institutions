# [ACTIVE] Sanity-snapshot tests for AH training and simulation

## Goal

Add cheap, repository-wide regression tests that catch silent breakage of
the AH-training and simulation pipelines. Each tiny fixture config is
run end-to-end on Raven with sanity instrumentation enabled, the resulting
log file is fetched back to local, and `pytest` (running locally) diffs
the fetched log against a checked-in "golden" snapshot.

The instrumentation surfaces config-derived invariants
(`n_player`, `n_groups`, `switch_every`, `seed`, `mask_name`,
`x_encoding`, fold sizing, RNG state, switch-round indices, model
shape, autoregression order, etc.) so any change in those flows lights
up in the diff.

## Why

Today the only test coverage on `src/aimanager/` is `test_encoder.py`
and `test_environment.py` — both unit-level. There is nothing
end-to-end. The 50-episode retrain (#85) and the FI work (#83/#84) both
moved enough surfaces that a harness-level snapshot test would have
flagged regressions cheaply.

Snapshot tests are also the right shape for this codebase because the
"correct" behaviour is mostly defined by the YAML config, not by code
invariants — i.e. the test answers "did the config flow through to the
model and data the way it says it should?".

## Design

### Three-stage flow

1. **Run on Raven** (skill: `/run-sanity-runs`) — submits the four tiny
   fixtures via the existing SLURM pipeline with `AIM_SANITY_LOG=1` and
   `AIM_SANITY_LOG_FILE=<path>` set, so each run writes a JSONL log to a
   known artifact path on the cluster.
2. **Fetch** (existing skill: `/fetch-cluster`) — pulls the log files
   back to local under `src/aimanager/tests/fixtures/_logs/` (gitignored).
3. **Test locally** (`pytest`) — pure-Python test code that opens the
   fetched `*.actual.jsonl` and the committed `*.golden.jsonl`, applies
   normalisation + float tolerance, and diffs. No torch / PyG imports in
   the test path, so tests run natively on macOS.

### Emission API

`src/aimanager/utils/sanity.py` exposes a single function:

```python
def emit(name: str, value, **ctx): ...
```

Controlled by `AIM_SANITY_LOG` env var (default off → emit() is a no-op
return). When on:
- If `AIM_SANITY_LOG_FILE=<path>` is set, append one JSON line per call
  to that file.
- Otherwise, print to stdout.

Format: `[SANITY] {"name": "...", "value": ..., "ctx": {...}}`. The
`[SANITY]` prefix lets the test parser ignore unrelated stdout noise
when stdout fallback is used; in file mode the parser reads the whole
file directly.

### Why this shape

- Call sites are one-liners (`emit("dataset.n_episodes", len(eps))`),
  so production code stays uncluttered.
- Default off → zero overhead, zero log noise outside tests.
- Adding new sanity checks doesn't require touching test code — just
  add an `emit()` and re-record the golden.
- A failing snapshot is *informative* — the diff itself tells you what
  changed, no expected-value lookup needed.
- Assertion logic lives entirely in golden files, not in production code.
- Tests are pure log-diffing and don't import the pipeline; they run
  natively on macOS without torch/PyG.
- The cluster run is a one-shot per behaviour change — no permanent
  CI wiring needed.

### Instrumentation points

Each point is one `emit()` call. Order matters — golden files are
sequence-sensitive.

**Training (`train_ah`):**

1. `config.loaded` — the resolved config dict (after grid expansion if any)
2. `dataset.n_episodes`, `dataset.n_rounds_per_episode`, `dataset.n_player_per_round`
3. `dataset.n_groups`, `dataset.switch_rounds` (derived from `switch_every`)
4. `dataset.x_features` — feature names actually fed into the model
5. `dataset.mask_name`, `dataset.mask_fraction_valid`
6. `cv.fold` (per fold), `cv.n_train_episodes`, `cv.n_test_episodes`,
   `cv.train_episode_ids_hash`, `cv.test_episode_ids_hash`
7. `model.spec` — `{y_levels, hidden_size, has_rnn, has_edge_model,
   has_global_model, n_params}`
8. `train.epochs_planned`, `train.batches_per_epoch`,
   `train.first_seed_state` (hash of torch RNG after seeding)
9. `train.final_test_loss_per_fold` (after the run completes)
10. `autoreg.min_predicted`, `autoreg.max_predicted` (when applicable)

**Simulation (`simulate`):**

1. `sim.config_loaded`, `sim.model_paths_resolved` (assert each model
   file exists and loads)
2. `sim.n_episodes`, `sim.n_rounds`, `sim.n_agents`,
   `sim.initial_agent_groups`, `sim.switch_every`
3. Per episode (sampled — only first 2 of N to keep golden small):
   - `sim.episode.start_groups`
   - `sim.episode.switch_rounds_observed` (assert all `% switch_every == 0`)
   - `sim.episode.group_size_per_round`
4. `sim.aggregates.contribution_mean`, `sim.aggregates.punishment_mean`,
   `sim.aggregates.common_good_mean`

### Fixtures

Three tiny training configs + one tiny sim config under
`src/aimanager/tests/fixtures/`. Each is a stripped copy of its
canonical 50ep ancestor with:

- `n_cross_val: 2` (instead of 5)
- `epochs: 2` (training) / `n_episodes: 4` (sim)
- `fraction_training: 0.1`
- A small fixed seed
- Output dir under `tests/fixtures/_artifacts/` (gitignored)

Files:

- `tests/fixtures/tiny_contribution.yml`
- `tests/fixtures/tiny_punishment.yml`
- `tests/fixtures/tiny_switch.yml`
- `tests/fixtures/tiny_simulation.yml`

Each fixture has a sibling golden:

- `tests/fixtures/tiny_contribution.golden.jsonl`
- ... etc.

### Test files

Pure-Python comparators — no pipeline imports. Each test reads the
fetched `*.actual.jsonl` from `src/aimanager/tests/fixtures/_logs/` and
diffs against its sibling `*.golden.jsonl` under
`src/aimanager/tests/fixtures/`.

- `src/aimanager/tests/test_sanity_training.py` — one parametrised test
  per AH target.
- `src/aimanager/tests/test_sanity_simulation.py` — same shape for sim.

Comparison utility lives in `src/aimanager/tests/_sanity_diff.py`:

- Parses both files line-by-line into structured records.
- Compares names + ctx exactly; values via float tolerance:
  `atol=1e-4` for shape/loss values, `atol=1e-3` for
  `train.final_test_loss_per_fold`, exact match for hashes / counts /
  shapes.
- Produces a unified diff on failure.

If `_logs/` is empty (i.e. the user hasn't run + fetched yet), the
test fails with a clear hint to invoke `/run-sanity-runs` and
`/fetch-cluster`.

### Golden update workflow

After a behaviour change is merged that changes a sanity emission:

1. Invoke `/run-sanity-runs` to re-run the fixtures on Raven.
2. `/fetch-cluster src/aimanager/tests/fixtures/_logs/`.
3. `pytest` to confirm the diff is the expected one.
4. If expected, copy each `_logs/<name>.actual.jsonl` over its
   corresponding `<name>.golden.jsonl` and commit.

A `pytest --update-goldens` flag is provided as a convenience that
performs step 4 automatically (after step 2).

## Plan steps

| # | Step | Optional |
|---|------|----------|
| 1 | Add `src/aimanager/utils/sanity.py` with `emit()` + env-var toggle (file or stdout) | No |
| 2 | Wire `emit()` calls at the listed instrumentation points in train + sim | No |
| 3 | Add 4 tiny fixture configs under `src/aimanager/tests/fixtures/` | No |
| 4 | Add `scripts/run_sanity_fixtures.sh` + `.claude/skills/run-sanity-runs/SKILL.md` to drive the cluster runs | No |
| 5 | Run all fixtures via the new skill, fetch logs locally, hand-validate, save as goldens | No |
| 6 | Add `_sanity_diff.py` comparator + `--update-goldens` pytest flag | No |
| 7 | Add `test_sanity_training.py` and `test_sanity_simulation.py` (pure Python, no torch import) | No |
| 8 | Update `.gitignore` to exclude `src/aimanager/tests/fixtures/_logs/` | No |
| 9 | Document the workflow in `CLAUDE.md` under Testing | No |

## Decisions locked in

- **Hash strategy for episode-id sets**: plain
  `hashlib.sha256(",".join(sorted(map(str, ids))))`, no version prefix.
  If the algorithm changes, re-record goldens — same workflow as any
  other behaviour change.
- **Float tolerance**: `atol=1e-4` for shape/loss values that should be
  exact-ish; exact match for shapes / counts / hashes.
- **`train.final_test_loss_per_fold` recording**: 4 significant figures,
  compared with `atol=1e-3` to absorb PyTorch patch-release wobble.

## Out of scope

- Unit tests for individual helper functions
- GPU-only paths (fixtures use `device: cpu`)
- Performance/benchmark snapshotting
- Snapshotting predicted *values* (we snapshot summaries — losses, not
  per-sample logits — to stay deterministic and small)

## References

- Existing tests: `src/aimanager/tests/test_encoder.py`,
  `test_environment.py` (run remotely via `scripts/remote_test.sh` —
  this work runs locally and is independent of that flow)
- Existing skills the new flow leans on: `/train`, `/simulate`,
  `/fetch-cluster`
