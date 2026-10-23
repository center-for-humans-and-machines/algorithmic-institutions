# Evaluation suite: from a simulation to scores

Handover sub-report 3. Scope: `src/aimanager/evaluation_suite/`, which turns a
finished simulation into the score rows the autoresearch campaign ranked stacks
by. Refs are to `yannik/handover` (HEAD `bb414f5a`); the suite, its tests and
`notes/eval*`/`evaluation_metric_defs.md` have no uncommitted changes there.
Campaign refs are to `origin/autoresearch` (`cd227b01`, 2026-09-23).

## TL;DR

- `python -m aimanager evaluate <sim config>` reads `<output_dir>/per_round.parquet`
  and writes `evaluation/{metrics.csv, scores.csv, visuals/*.jpg}` beside it.
  It needs no torch or PyG. It runs in this Linux container and reproduced the
  committed reference scores to 7e-16 (4 min 14 s, one pairing).
- A score is `mean_r d(h_a, s) / mean_r d(h_a, h_b)` over 500 fixed resamples
  (seed 42) of 25-vs-25 episodes. 1 is the human-vs-human noise ceiling.
- **21 vs 22 rows:** the code has 22 rows today. The atlas corpus (PRs #146–#181)
  was scored with 21: no RCE, and RCB had no `{0}` bin. Both changed on
  2026-09-24. On the same parquet the #179 tip's RCB went 2.315 -> 3.162. Old
  and new RCB scores cannot be compared.
- The "stack mean score" is the unweighted mean of the 21 `score` values of one
  pairing. No suite code computes it: agents wrote it into their PR results
  tables, and the summary scripts re-derive it as `sum/len`.
- Every campaign verdict is one training run, one sim (seed 42, 100 episodes)
  and one evaluation, and changing seeds was illegal. Measured noise: the
  21-row mean moves by SD about 0.08 across sim seeds (range about 0.2) and
  0.047 across training seeds. Rows ≤ 1 moves from 4 to 14. Only the large
  band jumps in the atlas are clearly real.

## Score rows

Slot = the campaign's attribution (`score_progressions.py` on `origin/autoresearch`).
Kind: **EMD** = 1-Wasserstein on raw samples; **abs Δ** = weighted mean
|human − sim| per stratum; **strat. EMD** = weighted mean of per-stratum EMDs.
C/S/P strata are weighted uniformly. R strata are weighted by human frequency,
fixed once on the full human data (`metrics.py:84-91`).

| id | slot | kind | measures |
|---|---|---|---|
| CA | contribution | EMD | per-participant mean contribution (diag: signed Δ std) |
| CB | contribution | abs Δ, 24 rounds | round-mean contribution trajectory |
| CC | contribution | EMD | group mean per (game, round) (diag: Δ std) |
| CD | contribution | EMD | all raw contributions pooled |
| CE | contribution | EMD | group 0 − group 1 mean per (game, round) (diag: Δ std) |
| CF | contribution | abs Δ, 48 cells | share at 0 and at 20, per round |
| CG | contribution | abs Δ | spread ratio: SD(group means) / SD(individual) |
| RCA | contribution | strat. EMD, 4 types | Δc after no-switch / switched / chose-stay / stayed-but-comp-changed |
| RCB | contribution | abs Δ, 5 bins | mean Δc of non-full contributors by rate p/(20−c); `{0}` bin since 09-24 |
| RCC | contribution | abs Δ | full contributors: Δc punished − Δc unpunished |
| RCD | contribution | abs Δ | OLS slope of switchers' Δc on (receiving-group mean − own c) |
| RCE | (not in campaign) | abs Δ, 4 bands | OLS slope of Δc on punishment within c bands 0-4/5-9/10-14/15-19 |
| SA | switch | abs Δ | switch rate over valid opportunities |
| SB | switch | abs Δ, 5 rounds | switch rate per decision round 3/7/11/15/19 |
| SC | switch | EMD | larger-group size per (game, round), rounds ≥ 4, empty groups kept |
| RSA | switch | abs Δ, 3 bins | switch share of punished players by punishment 1-3/4-15/16+ |
| PA | punisher | EMD | all received punishments, zeros included |
| PB | punisher | abs Δ, 24 rounds | round-mean punishment |
| PC | punisher | abs Δ, 24 rounds | share of zero punishments per round |
| PD | punisher | abs Δ | spread ratio of group-mean punishment |
| RPA | punisher | strat. EMD, 6 bins | punishment distribution by contribution bin {0},1-5,…,{20} |
| RPB | punisher | strat. EMD, 3 bins | punishment distribution by group size 1-3/4-5/6-8, rounds ≥ 4 |

Row families live in `metrics.py`: C `:146`, S `:207`, P `:241`, R `:280`
(KINDS `:286-295`), registry `GROUPS` `:509`. The definitions match
`notes/evaluation_metric_defs.md`. Two stale bits in that note: the header
at `:137` still says "implemented on a later branch", and `:22` says the sim
has no timeouts. That is still true for switching, but sims have recorded
contribution timeouts since 2026-09-29.

## 1. Pipeline

**Canonical frame** (`convert.py:4-18`, schema `CanonicalAgentRound` `:54`,
validated by pandera in `_finalize` `:92`). One row per (episode, participant,
round): `episode_id, participant_code, round_number, group_id, contribution
(NaN = no input), punishment (NaN = no input), does_switch, switch_mask,
switch_valid`. `common_good` and `payoff` are dropped. `_derive_switching`
(`:66`) labels the decision round: rounds with `(t+1) % switch_every == 0`
and a following round, i.e. 3, 7, 11, 15, 19.

**Human** (`load_human`, `:111`). The source is `experiments/2group_8agent_50ep.csv`
(`:38`): 100 episodes, 19,200 rows. The file is flip-doubled. Keeping the
lower `episode_id` per `pair_id` (`:114-116`) leaves 50 games, 9,600 rows.
`contribution` is NaN where `player_no_input` (280 rows), `punishment` NaN
where `manager_no_input` (407 rows) (`:117-118`). `switch_valid` drops
decision rows whose next-round `selection_timeout` is set (1,891 of 2,000
remain).

**Sim** (`load_sim`, `:123`). Returns one frame per `run` (pairing) in the
parquet. Since `7c070cf7` (2026-09-29) it blanks `contribution` where
`contribution_valid` is False (`:130-132`). The frontier sim has 1.8 %
timeouts; humans have 2.9 %. Campaign-era parquets have no such column, so
nothing is blanked there. Sim punishments are never NaN. `switch_valid` equals
`switch_mask`, because the sim has no `selection_timeout`. `timeout_contribution`
(`simulate.py:195`, validated in `manager/environment.py:89-98`; 0-20 or
`default` = dataset median) sets the value the models see for a timed-out
player. The evaluation blanks that value, so the setting reaches scores only
through the dynamics.

**Discrepancy** `MetricGroup.d` (`metrics.py:93-137`). EMD via
`scipy.stats.wasserstein_distance` (`:107`). An empty weighted stratum raises
`ValueError` (the #134 guard). `evaluate()` (`evaluate.py:38`) writes raw `d`,
`std_diff` for CA/CC/CE (`:35`) and `n_human`/`n_sim` to `metrics.csv`.

**Noise-ceiling score** (`scoring.py`; `notes/eval_scoring_schema.md`).
`make_repeats` (`:26`) uses one `default_rng(seed)`. For each of R repeats it
permutes the 50 human ids into halves h_a/h_b (m = 25) and draws 25 of the
sim's episodes without replacement. `score_row` (`:81`) averages numerators
and denominators separately over surviving repeats, then divides. A repeat
whose subset lacks a stratum is dropped. If fewer than R/2 survive, the score
is NaN with a warning (`:112`). The denominator is shared across pairings
(`score_all`, `:122`), and all pairings must share one episode pool (`:129`).
R = `scoring_repeats` (500), seed = `scoring_seed` (42) (`evaluate.py:94-95`).
The campaign froze both. The plan is identical for every candidate, so
re-scoring a parquet is deterministic. On the #179 parquet, every row matched
exactly except RCB and the new RCE. Bands: ≤ 1 ceiling, 1-2 minor, 2-5 clear,
> 5 not reproduced.

**Stack mean.** No suite or sweep code computes it. The rule is
`notes/autoresearch.md:28-31` (gate 2: the mean over all 21 rows may rise at
most 10 % above baseline), and agents logged the value in results tables
(`:298`). On `origin/autoresearch`,
`scripts/data_analysis/autoresearch_summary/score_progressions.py:110` and
`leaderboard.py:132` recover each PR's `scores.csv` vector by matching
`sum(scores)/len(scores)` to the logged mean. `:85` keeps only runs with all
21 metrics. So the stack mean is one pairing's `score` column, unweighted,
over 21 rows. A 23-family `scores.csv` holds four punisher pairings; filter
`run` first.

## 2. Entrypoints and outputs

```bash
cd /opt/algorithmic-institutions          # HUMAN_DATA_FILE and output_dir are cwd-relative
python -m aimanager evaluate configs/simulation/manager_testing/<sim>.yml
python scripts/data_analysis/evaluation_sweep.py <name> plots/simulation/23_..._self_<c>_contr_<s>_switch ...
python -m pytest src/aimanager/tests/test_eval_*.py -q
```

- `cli.py:42` requires `artificial_humans, managers, n_episodes,
  n_episode_steps`. No model is loaded. `run_cli` (`evaluate.py:63`) uses
  `output_dir`, or the config path without `.yml` (`:65-67`), and exits 1 with
  a `save_per_round` hint if the parquet is missing (`:69`). The sim must set
  `save_per_round: true` (`simulate.py:842`). Optional keys: `switch_every`
  (4), `scoring_repeats`, `scoring_seed`. There is no output flag. To write
  elsewhere, copy the parquet and point a config copy's `output_dir` at it
  (that is how I ran it below).
- Outputs: `metrics.csv` (run, metric, kind, d, std_diff, n_human, n_sim);
  `scores.csv` (run, metric, score, numerator, denominator, repeats_used,
  n_repeats, seed); `visuals/` with 25 JPGs. SA has no figure; CE, CG, PD and
  SC have two each (`visuals.py:616`, `matplotlib.use("Agg")` at `:19`).
- `evaluation_sweep.py` parses `_self_<contr>_contr_<switch>_switch$`
  (`:90`) and pivots to `plots/data_analysis/evaluation/<name>/score_matrix.csv`
  (`:119`, `:345`), plus `slot_report.jpg`, `slot_concordance.jpg` and
  `switch_unrolled.jpg`. **`METRIC_ORDER` (`:33`) lists 21 rows without RCE,
  so the matrix silently drops RCE.** It has no mean column.
  `23_stack_sweep_updated/score_matrix.csv` is 32 stacks x 21 rows, old RCB.

## 3. Data flow

```
experiments/2group_8agent_50ep.csv --load_human (dedupe pair_id, NaN no-input)--+
                                                                                 +--> canonical frames
plots/simulation/<sim>/per_round.parquet --load_sim (per run, blank timeouts)---+         |
   (simulate with save_per_round: true)                                                  v
          metrics.GROUPS[C,S,P,R].d(human, sim)  ---------------------------> evaluation/metrics.csv
          scoring.score_all: 500 x (h_a, h_b, s25), seed 42 -> num/den ----> evaluation/scores.csv
          visuals.plot_all --------------------------------------------------> evaluation/visuals/*.jpg
scores.csv (one run) -> mean over 21 rows = "stack mean" (logged by hand; summary scripts re-derive)
many scores.csv -> evaluation_sweep.py -> score_matrix.csv (21 cols) + slot figures
```

## 4. Three metric versions

| version | where | rows | RCB | sim timeouts |
|---|---|---|---|---|
| v21 | atlas corpus #146–#181, `origin/autoresearch` | 21 | 4 punished bins | not recorded |
| v22-oldRCB | later campaign branches (`metrics/rce-response-slope` 09-21, `auto/seed-spread-noise-floor`) | 22 | 4 bins | not recorded |
| current | main / `yannik/handover` since `aa0da68c` and `194f6ed7` (09-24) | 22 | `{0}` + 4 bins | blanked since `7c070cf7` (09-29) |

CLAUDE.md's "22 rows" is correct for the code. The atlas's "21" and its
"verbatim" RCB text come from v21. On #179's unchanged parquet, the v21 21-row
mean was 1.0988 and the current one is 1.139, all of it from RCB. The
noise-floor log reports one trap: an editable install pointing at another
checkout silently scored 21 rows. Here `aimanager` resolves to
`/opt/algorithmic-institutions/src`.

## 5. Seed noise and reading the atlas

The campaign used no seed replicates. The protocol is one sim at seed 42 with
100 episodes and one evaluation (`notes/autoresearch.md:42-48`, `:160`). It
forbids "changing or shopping simulation / scoring seeds" (`:103`, `:180`).
The atlas confirms "24 rounds x 100 episodes, seed 42". Two later measurements
exist:

- **Sim seed** (same models, `seed: 42..46`). Seed configs are in
  `configs/simulation/manager_testing/ablation_224/seeds/` and the seed-42
  configs are the FRONTIER and abl0 configs. I computed this from the
  committed `scores.csv`. Family `curpun_t0`: per-row SD median 0.16, CA/CB/CD
  0.29-0.32, RPA 0.02. 21-row mean per seed 1.290/1.146/1.190/1.200/1.075 (SD
  0.079, range 0.215). Rows ≤ 1: 6-11. Family `abl0_old_tdefault`: median SD
  0.10, mean 1.127-1.315 (SD 0.080). Seed 42 was the lowest of five, and rows
  ≤ 1 ran 4 to 13. The `abl3_noflag_t0` seed dirs have no `evaluation/` yet.
- **Training seed** (sim seed fixed at 42). From
  `origin/auto/seed-spread-noise-floor:notes/autoresearch_log/seed-spread-noise-floor.md`,
  unmerged, six contributor retrains on the PR #192 stack, v22-oldRCB.
  Per-row median SD 0.138. 22-row mean SD 0.047, range 0.117. Rows ≤ 1 SD
  3.16, range 6-14. Ten of 22 rows sit within one SD of a band edge, so a
  single run cannot gate them: CA, CB, CC, CD, CF, CG, SA, SB, RCE, RSA. The
  shipped baseline was the best of six on every aggregate, which suggests
  selection.

How to read the 13 successes and 23 failures:

- These moves are far outside both noise sources and are real: CG 9.81 -> 4.16
  (#165), 5.91 -> 2.83 (#170), 4.27 -> 0.90 (#179); PD 2.93 -> 1.53 (#160);
  SC 2.19 -> 1.15 (#171), 2.67 -> 1.30 (#172); RCA 3.51 -> 1.86 (#177).
- #174 is within noise: SC 1.30 -> 0.98 lands 0.02 under the edge, against an
  SC SD of 0.13-0.18.
- Mean gaps below about 0.1 are not resolved by one seed. Examples: #179 1.099
  vs #177 1.125, #179 vs #181 1.105, #172 -> #174 0.039. So the ordering of the
  two tips is unknown.
- A failure decided by a near-miss on a boundary row may also be noise.
- Rows ≤ 1 (used to pick the evaluation stack, §3 of the guideline) is the
  noisiest summary.
- To compare stacks, run at least 3-5 sim seeds for each stack and compare
  per-row means ± SD. Do it for the baseline as well as the candidate.

## 6. Where it runs

- **This container:** Linux x86_64, Python 3.10.12, torch 2.5.1+cpu,
  **PyG 2.6.1 present**, pandas 2.2.3, pandera 0.34.1, scipy 1.14.1,
  matplotlib 3.7.3, pyarrow 25.0.1, numpy 1.26.4. The whole suite runs here.
  "Local macOS" in CLAUDE.md (`:257`) is historical. The code has no
  darwin/mps/`/Users` reference, and plotting uses the Agg backend.
- **Path and cwd assumptions:**
  - `HUMAN_DATA_FILE` and `SIM_EXAMPLE_FILE` are repo-relative strings
    (`convert.py:38-39`).
  - `run_cli` opens the human data relative to the cwd. Run from the repo
    root. `test_run_cli_writes_metrics_and_scores` also depends on this.
  - Sim configs use relative `output_dir: plots/simulation/...`.
  - The sweep writes to a cwd-relative `plots/data_analysis/evaluation/`.
  - The tests read the real human CSV and the LFS-tracked
    `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet`.
  - `git-lfs` is not installed here. Working-tree LFS files are real, but
    older versions (`git show <ref>:plots/...`) are pointers that cannot be
    fetched offline.
  - Without the LFS clean filter, `git status` reports every LFS-tracked file
    in the working tree as modified. I checked all 260 `M` entries on this
    branch (2026-10-23): each file's sha256 equals its committed pointer's
    oid. Most of the "uncommitted changes" are therefore not real edits.
- **Raven image:** not needed and not tested. `evaluate` is CPU-only and
  single-threaded. The image's `requirements.raven.txt` pins
  `pandera>=0.18,<0.23` and `scipy<1.14`, while `pyproject.toml:13` asks for
  `pandera>=0.26.1`. The proxy's documented `program_call`s cover only
  `train-ah|train-manager|simulate`. Simulate on Raven, sync back
  `per_round.parquet`, evaluate here.

## 7. Results of what I ran

- `python -m pytest src/aimanager/tests/test_eval_*.py -q` from the repo
  root: **73 passed, 48 warnings in 27.28 s**, 0 failed and 0 errors.
  Warnings: 47 `PyparsingDeprecationWarning` from matplotlib, plus 1 pandera
  `FutureWarning` ("Importing pandas-specific classes and functions from the
  top-level pandera module will be removed"; fix: `import pandera.pandas as pa`).
- `python -m aimanager evaluate` on a copy of
  `24_FRONTIER_vnode_curpun_self_gnncopar1_contr_gnn_switch`. Outputs are in
  `/tmp/claude-1000/-opt-algoinst-proxy/7dd273a2-cb59-40d9-9e03-c1e69d7fca2f/scratchpad/eval_scout/frontier/evaluation/`.
  Result: 22 metrics, 22 scores, 25 figures, 4 min 14 s wall time. Max |Δ| vs
  the committed files: `scores.csv` 6.7e-16, `metrics.csv` 9.7e-17.
  `repeats_used` was 500/500 on every row. 22-row mean 1.292, 21-row mean
  1.290, rows ≤ 1 6/21. Nothing in the repo was written.

## Open questions

1. Should RCB-v21 be kept, so atlas numbers stay reproducible? Or should the
   atlas spine be re-scored with current rows? Re-scoring needs the
   campaign-branch parquets, which are LFS objects and not available here.
2. Should "stack mean" become 22 rows, and be computed in code, for example a
   `mean`/`rows_le1` column in `evaluation_sweep.py`? Its hard-coded 21-row
   `METRIC_ORDER` drops RCE today.
3. Should the protocol require multiple sim seeds, and training seeds, for
   baseline and candidate alike? If so, how many, and should
   `seed-spread-noise-floor`'s per-row SDs be merged as reference error bars?
4. Should the uniform-vs-human-frequency weighting text for RSA/RPA/RPB in the
   defs note be stated per row? Code uses human-frequency weights for all R
   rows.
5. Should `HUMAN_DATA_FILE` be resolved relative to the package rather than
   the cwd, so `evaluate` works from any directory?
