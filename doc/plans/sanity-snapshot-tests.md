# [ACTIVE] Sanity-snapshot tests for AH training and simulation

## Goal

Add cheap, repository-wide regression tests that catch silent breakage of
the AH-training and simulation pipelines. Each test runs a tiny fixture
config end-to-end with sanity instrumentation enabled, captures the
emitted log, and diffs it against a checked-in "golden" snapshot.

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

### Two-layer instrumentation

1. **Emission** — `src/aimanager/utils/sanity.py` exposes a single
   function:

   ```python
   def emit(name: str, value, **ctx): ...
   ```

   Controlled by `AIM_SANITY_LOG` env var (default off → emit() is a
   no-op return). When on, prints one JSON line per call:
   `[SANITY] {"name": "...", "value": ..., "ctx": {...}}`.

2. **Assertions** — there are none in production code. All
   "expected behaviour" lives in golden log files. Tests compare
   captured stdout to the golden after a normalisation pass that strips
   timestamps and other non-deterministic noise.

### Why this shape

- Call sites are one-liners (`emit("dataset.n_episodes", len(eps))`),
  so production code stays uncluttered.
- Default off → zero overhead, zero log noise outside tests.
- Adding new sanity checks doesn't require touching test code — just
  add an `emit()` and re-record the golden.
- A failing snapshot is *informative* — the diff itself tells you what
  changed, no expected-value lookup needed.
- All sanity-check semantics live in `sanity.py` + golden files, not
  scattered through pipeline code.

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

- `src/aimanager/tests/test_sanity_training.py`:
  one parametrised test per AH target. Each runs the fixture via
  subprocess with `AIM_SANITY_LOG=1`, captures stdout, normalises, and
  diffs against the golden.

- `src/aimanager/tests/test_sanity_simulation.py`: same shape for sim.

Comparison utility lives in
`src/aimanager/tests/_sanity_diff.py`:

- Strips lines without the `[SANITY]` prefix
- Normalises numeric fields with a tolerance (`atol=1e-4`) for floats
  that legitimately wobble (param counts are exact; losses use atol)
- Produces a unified diff on failure

### Golden update workflow

A pytest flag `--update-goldens` re-records all golden files. Reviewer
sees the resulting git diff in the same PR that introduced the
behaviour change. Without the flag, mismatches fail the test.

### Why subprocess

Running the pipeline in-process risks bleeding torch/numpy RNG state
across tests and hides import-time side effects. A subprocess is also
the most realistic regression test — it catches CLI breakage too.

## Plan steps

| # | Step | Optional |
|---|------|----------|
| 1 | Add `src/aimanager/utils/sanity.py` with `emit()` + env-var toggle | No |
| 2 | Wire `emit()` calls at the listed instrumentation points in train + sim | No |
| 3 | Add 4 tiny fixture configs under `src/aimanager/tests/fixtures/` | No |
| 4 | Add `_sanity_diff.py` comparison helper + `--update-goldens` pytest flag | No |
| 5 | Manually run each fixture once, hand-validate the emitted log, save as golden | No |
| 6 | Add `test_sanity_training.py` and `test_sanity_simulation.py` | No |
| 7 | Wire into `scripts/remote_test.sh` (no extra flags needed — they run as part of `pytest`) | No |
| 8 | Document the workflow in `CLAUDE.md` under Testing | No |

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
  `test_environment.py`
- Cluster test runner: `scripts/remote_test.sh`
- Example of fixture-driven test logs already used:
  `scripts/tests/test_remote_test.py`
