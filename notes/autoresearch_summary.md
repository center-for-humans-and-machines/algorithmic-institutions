# Autoresearch Summary: Reporting the Campaign

A standing guideline for building the summary of the autoresearch campaign
(`notes/autoresearch.md`): one interactive report covering every experiment
PR. The framework's rules changed over time; this document reports against
what each experiment was judged by at its time, and never re-judges.

---

## 1. Scope and sources

- **Corpus**: every autoresearch experiment PR, #146 onward, `[SUCCESS]`
  and `[FAIL]` alike. Failures are half the story and are never dropped.
- **Sources of truth**, in order: the PR (title, body, base/head branches,
  dates) and the head branch's `notes/autoresearch_log/<slug>.md`
  (declaration, results table, notes). `main`'s log dir only holds merged
  logs, so collection always goes through the head branches via `gh`.
- Scores enter the summary exactly as they appear in the results tables —
  no re-running, no re-rounding, no re-judging by later criteria.

## 2. Everything is a script

No number, node, or ranking is ever hand-typed into the report. The
pipeline lives in `scripts/data_analysis/autoresearch_summary/`:

| step | script | output |
|---|---|---|
| collect | `collect.py` | `data/experiments.json` |
| classify | `classify.py --prepare / --merge` (blinded input; validated assignments) | `data/categories.json` |
| tree | `progress_tree.py` | `.../progress_tree.png` |
| spine scores | `score_progressions.py` (fetches spine PRs' scores.csv via LFS; caches) | `data/spine_scores.json`, `.../score_progressions.png` |
| breakdown | `score_breakdown.py` | `.../score_breakdown.png` |
| build | `build_report.py` (interactive report; machinery view) | `plots/data_analysis/autoresearch_summary/report.html` |

Outputs are regenerated end-to-end from the scripts; editing an output by
hand is illegal. Intermediate `data/*.json` files are committed so the
report is reproducible without GitHub access.

## 3. The data model

One record per PR in `experiments.json`:

- `pr`, `title`, `verdict` (SUCCESS/FAIL), `created_at`, `merged`
- `base_branch`, `head_branch`, `parent_pr` (resolved from base branch;
  `null` means main)
- `slot` (contribution / switch / punisher), from the declaration
- `declaration_method`: the declaration's method description (the first
  part, before targets/plan) — the only text classification may see
- `baseline` and `result`: `{target_rows: {row: score}, mean, rows_le_1}`
  from the results table, plus `band_steps`: total bands gained across
  declared target rows

## 4. Classification

- Done by **Sonnet** agents for cost; the orchestrator only spot-checks.
- Input per PR: `declaration_method` only — not the results, not the
  verdict, not the title. Method is classified by what was tried, never by
  how it went.
- Fixed taxonomy (an agent must pick exactly one; `other` is a finding,
  not a dumping ground — three or more `other`s means the taxonomy needs a
  new class, decided by the maintainer):
  1. `correlated-sampling` — copulas and shared latents applied at
     sampling time
  2. `persistent-latent` — training-time latent variables (per-agent or
     per-group types)
  3. `nonlinear-emission` — MLP / XGBoost / regression emission heads
  4. `autoregressive` — observed-history conditioning
  5. `structured-head` — joint or structured decision heads (transition
     structure, exodus heads)
  6. `feature-engineering` — new input features on an unchanged model
  7. `training-regime` — curriculum / sampling-schedule changes
  8. `architecture` — graph / attention structure changes
  9. `other`
- Output per PR: `{pr, category, rationale}` — one sentence of rationale.

## 5. The report

One self-contained HTML file (plotly, no server, no external assets),
`plots/data_analysis/autoresearch_summary/report.html`, with two views:

**Progress tree** (the centerpiece):

- Node = PR at x = chronological order; y = a **toggleable metric**:
  stack mean (default), rows <= 1, or the declared-target score. No single
  axis is privileged because the success criteria changed over time and
  the current rule lets the mean drift within a margin.
- Edge = child -> parent (main is the root, at the pre-campaign baseline).
  The two deep lineages (copula-on-GNN, gaussian-MLP) get distinct edge
  styles so the two trees read instantly.
- Color = category (§4); `[FAIL]` nodes are hollow/grey with a colored
  outline. A badge on the node marks a band upgrade — the decisive
  criterion lives on the node, never on the axis.
- Hover = PR number, title, hypothesis one-liner, baseline -> result on
  the target rows, mean, rows <= 1, category rationale.
- A frontier step-line traces the best stack mean reached so far.

**Leaderboard**:

- One row per PR, ranked against **its own baseline** (main's or its
  parent's — stacked PRs are judged on their own increment).
- Three toggleable ranking criteria, matching the framework's §2 metrics:
  `band_steps` on declared targets, relative mean change, and
  Δ rows <= 1. The reader picks the sort; the default is band_steps,
  then mean, then rows <= 1 as tie-breakers — the framework's own order.
- Successes and failures in separate tables; failures carry a
  "closest miss" column (distance to the nearest band edge).

## 6. Conventions

- All work happens on the dedicated branch (`autoresearch-summary`), one
  PR into main at the end.
- Scripts run locally (macOS): `gh` for collection, plotly for the build.
  Nothing here touches Raven or the frozen surface of
  `notes/autoresearch.md` §8.
- The report never contradicts a PR's recorded verdict. Where an old
  verdict would flip under today's rules (e.g. the 10% mean margin), the
  tooltip may note it; the node keeps its historical verdict.
- Prose findings (the narrative document around the visuals) live in
  `reports/autoresearch_summary.md` and cite PR numbers, never bare
  scores without their PR.
