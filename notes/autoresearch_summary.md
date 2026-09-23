# Autoresearch Summary: Reporting the Campaign

A standing guideline for building the summary of the autoresearch campaign
(`notes/autoresearch.md`): one interactive report covering every experiment
PR. The framework's rules changed over time; this document reports against
what each experiment was judged by at its time, and never re-judges.

**The deliverable** is a single self-contained file,
`plots/data_analysis/autoresearch_summary/report_bundle.html`, regenerated
end-to-end by the pipeline in §2 and published as a Claude artifact by
that pipeline's last step. It is **generated, not tracked** — see the
regeneration note below. Re-running the pipeline updates the same
artifact in place, so its URL — and any share link served from it —
stays valid.

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
pipeline lives in `scripts/data_analysis/autoresearch_summary/` and runs
in this order (each step idempotent; `PYTHONPATH` set to the pipeline dir
so the scripts import each other):

| # | step | script | output |
|---|---|---|---|
| 1 | collect (FROZEN) | `collect.py` | `data/experiments.json` |
| 2 | classify (FROZEN) | `classify.py --prepare / --merge` (blinded input; validated assignments) | `data/categories.json` |
| 3 | spine scores | `score_progressions.py` (fetches spine PRs' scores.csv via LFS, identified by mean-matching; also home of the shared metric/palette constants) | `data/spine_scores.json` |
| 4 | leaderboard | `leaderboard.py` (unique successes vs own baseline; 4 toggleable criteria; fetches missing vectors via LFS) | `data/leaderboard_scores.json`, `.../leaderboard.html` |
| 5 | before/after figures | `stack_visuals.py` (copies the reference stack's evaluation figures and fetches each tip's from its branch, sim dir found by mean-matching) | `data/stack_visuals.json`, `.../stack_visuals/{main,gnn,gmlp}/*.jpg` |
| 6 | machinery | `machinery.py` (drawn from the curated `data/stack_parts.json`; #PR pills link to plain-language story pages from `data/machinery_notes.json`, one page per unique method) | `.../machinery.html`, `.../machinery_pages/*.html` |
| 7 | build | `build_report.py` (multi-file working copy; `progress_tree.py` holds its tree-layout helpers) | `.../report.html` |
| 8 | bundle | `bundle_report.py` (THE deliverable: machinery, leaderboard and stories become in-page layers; figures inlined as data URIs) | `.../report_bundle.html` |
| 9 | publish | the Artifact tool, with the parameters step 8 prints (from `data/artifact.json`) | the artifact, updated in place |

**The rendered outputs are not committed.** `report_bundle.html` and its
siblings (`report.html`, `machinery.html`, `leaderboard.html`,
`machinery_pages/`) are gitignored: they rebuild byte-for-byte from the
committed inputs, and the bundle inlines 72 JPEGs as base64, so tracking
it would add ~5 MB of incompressible history per rebuild. One command
brings them all back:

```bash
scripts/data_analysis/autoresearch_summary/regenerate.sh
```

which runs steps 3-8 in order and ends by printing the step-9 publish
call. What *is* committed is everything those steps read: the
`data/*.json` caches and `stack_visuals/` — the latter deliberately,
because it is fetched from the experiment branches rather than
generated, and would be unrecoverable once those branches are deleted.

**Step 9 is the one step a shell cannot run.** Publishing goes through
the Artifact tool, so step 8 ends by printing the exact call to make.
Everything needed for it is committed in `data/artifact.json`: the
artifact `url`, the `favicon` (kept stable so readers keep finding the
tab), the title and the gallery description. Pass that `url` and the
existing artifact is updated; omit it and a duplicate is created
instead. To adopt a different artifact, point `url` at it; to start a
fresh one, set `url` to null and record what comes back. Share links
carry a secret key and are deliberately not committed.

Steps 1-2 are frozen with the corpus (#146-#181, maintainer ruling) —
never re-run them to pick up newer PRs. Steps 3-5 are fetch-and-cache:
their `data/*.json` caches and the fetched figures are committed, so
re-running steps 3-8 is offline-deterministic and reproduces the
deliverable byte-for-byte from the repo alone; step 9 then pushes that
byte-identical file to the artifact. Editing an output by hand
is illegal; curated inputs (`stack_parts.json`, `machinery_notes.json`)
are data files, edited there and only there.

## 3. The data model

One record per PR in `experiments.json`:

- `pr`, `title`, `verdict` (SUCCESS/FAIL), `state`, `created_at`
- `base_branch`, `head_branch`, `parent_pr` (resolved from base branch;
  `null` means main), `log_file`
- `slot` (contribution / switch / punisher) and `base_model`, from the
  declaration
- `declaration_method`: the declaration's method description (the first
  part, before targets/plan) — the only text classification may see
- `results_rows`: the results table's rows verbatim, and `metrics`: the
  confirmed run's `{mean, rows_le_1, target_scores_raw, verdict_raw}`
  (the last table row with a plausible mean)

## 4. Classification

- Done blinded by the summary agent itself (at 36 PRs, agent fan-out was
  judged unnecessary); `classify.py` provides the guard rails —
  `--prepare` strips verdict tokens from the input, `--merge` validates
  coverage, taxonomy membership and rationales.
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

Hand-authored SVG + vanilla JS, no server, no external assets. Two
copies: the multi-file working copy rooted at
`plots/data_analysis/autoresearch_summary/report.html` (machinery,
leaderboard and the story pages as sibling files), and the deliverable
`report_bundle.html` — one self-contained file in which those siblings
become in-page layers, the 72 before/after figures are inlined as data
URIs, and story links jump in place. The bundle opens offline from disk
and is what gets shared or published as a Claude artifact (republishing
the same file path keeps the artifact URL stable).

**Progress tree** (the welcome layer):

- Node = PR at x = chronological order, y = the stack mean it left
  behind (log scale, lower is better); main is the root at the
  pre-campaign baseline.
- Edge = child -> parent. The two deep trees carry distinct colors;
  solid edges are the success spines, dashed edges failed branches,
  faint edges main one-offs. A dotted step traces the best mean so far.
- Color + marker shape = category (§4); `[FAIL]` nodes faint grey
  (hollow when the run produced no evaluation).
- Hover = PR number, verdict, title, stack mean, category. Click = the
  machinery story page where one exists, the GitHub PR otherwise.

**Toggleable layers** in the same page: all 21 scores along the two
success spines (small multiples; every point hoverable and clickable
like a tree node); the per-spine score breakdown (21 family-colored
lines plus the bold mean; hover a line for its values, slot-focus
toggles in the corner); and the before/after layer — the evaluation
suite's own figure per score row for the reference stack and both
frontier tips side by side, navigable across rows (rows with two
figures show both pairs; SA has none). The nav links
out to the two sibling pages built by their own scripts —
`machinery.html` (whose #PR pills open the story pages under
`machinery_pages/`) and `leaderboard.html` (one row per unique success
vs its own baseline, four toggleable ranking criteria: Δ mean,
Δ rows <= 1, Δ rows > 2 reversed, band upgrades) — and both link back.

## 6. Conventions

- All work happens on the long-lived `autoresearch` branch, which holds
  the campaign's reporting and review work and is **not merged into
  `main`**. That keeps `main` free of this pipeline's gitignore rules and
  of the ~5 MB rebuilt bundle, and it is why the rendered outputs can
  simply be dropped (§2) rather than negotiated with `main`. Anything
  genuinely wanted on `main` is cherry-picked deliberately.
- Scripts run locally (macOS): `gh` + `git lfs` for collection and
  fetching, hand-authored SVG + vanilla JS for the build — no plotting or
  charting libraries. Nothing here touches Raven or the frozen surface of
  `notes/autoresearch.md` §8.
- The report never contradicts a PR's recorded verdict. Where an old
  verdict would flip under today's rules (e.g. the 10% mean margin), the
  tooltip may note it; the node keeps its historical verdict.
- Prose findings (the narrative document around the visuals) live in
  `reports/autoresearch_summary.md` and cite PR numbers, never bare
  scores without their PR.
