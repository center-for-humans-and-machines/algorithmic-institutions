# Autoresearch: Optimizing the Artificial-Human Stack

A standing guideline for research agents that improve the artificial-human
models (contribution / switch / punisher) against the evaluation suite. The
specific research campaign an agent runs on top of this lives in its own
document; this file defines the objective, the rules, and the loop.

---

## 1. Mission

Make the simulated stack indistinguishable from the human games. Progress is
measured by the evaluation suite only (`python -m aimanager evaluate`,
definitions in `notes/evaluation_metric_defs.md`). Everything you do must
cash out as a better score on it.

## 2. The metrics

Three numbers, judged in this order, all from one `evaluation/scores.csv`
(21 rows):

1. **Target rows** — the rows your declared hypothesis attacks (§6 gives you
   the candidates; e.g. the GNN contributor's are CG, RCA, RCD). You may
   target them collectively or a declared selection, but their scores must
   drop. No movement on the targets means the hypothesis failed — discard,
   whatever else happened to move.
2. **Rows <= 1** — the number of rows at or below the human-vs-human noise
   ceiling, for the whole stack. Must not fall; raising it is the program's
   headline number. **Baseline: 11 / 21** (reference stack, PR #143).
3. **Mean score** — the average over all 21 rows. Must not rise; breaks
   ties. **Baseline: 1.76.**

An experiment is **kept** iff its target rows improve and neither stack
metric regresses. It is a **headline improvement** when rows <= 1 rises.
The ordering is the anti-frankenstein guard in numeric form: you cannot buy
your target rows by quietly breaking the rest of the stack.

## 3. Evaluation protocol (two-stage)

The metric is a property of a full stack, so candidates are always scored
inside one.

**Reference stack** (current): `gnn` contribution x `gnn` switch x
`lin_multinomial` punisher — the winner of the 23-family sweep. Only the
human maintainer updates this definition, when a candidate is accepted.

- **Stage 1 — iterate.** Swap your candidate into its slot of the reference
  stack. One simulation (the standard protocol, see §7) + one evaluation.
  This is the number you iterate on.
- **Stage 2 — confirm.** A claimed improvement gets the full sweep: the
  8-config family with your candidate replacing its slot everywhere, then
  `evaluation_sweep.py`. The claim stands only if the candidate also wins its
  slot across contexts (best in most contexts on its slot's rows — the
  Kendall's-W discipline of PR #143), not just in the reference context.

## 4. Agents and slots

One slot per agent: **contribution**, **switch**, or **punisher**. You change
only your slot's model, features, and training configs. One change per
experiment, declared before you start: slot, base model, target rows (from
§6), hypothesis.

A bug fix in shared code (encoder, simulation, preprocessing) is legal but is
its own experiment: fix only, before/after scores for the whole reference
stack, no model change mixed in.

## 5. Legal and illegal changes

**Legal** — anything that plausibly makes the model more *human*, built on a
direction the evaluations point to (§6) or a finding you make and document:

- architecture changes,
- new input features — only information the real player or manager observably
  had at decision time (punishment models condition on round t-1, never on
  the current round's contributions),
- hyperparameter search, including selecting between variants by their
  Stage-1 score,
- training-data handling within the existing conventions (GNNs train on the
  flip-doubled data, linears on the single copy),
- bug fixes, with an explanation of what was wrong.

Every change carries a one-sentence behavioral rationale: *which* human
behavior it captures and *which* metric row that should move. If you cannot
write that sentence, the change is a frankenstein — do not make it.

**Illegal** — anything that improves the number without improving the model:

- touching the frozen surface (§8),
- changing or shopping simulation / scoring seeds and episode counts,
- features engineered at a metric's definition rather than at behavior
  (e.g. anything keyed to a specific bin edge or stratum boundary),
- training on the evaluation's resampling structure, or on the flipped
  duplicates where the convention says single-copy,
- reporting Stage-1 numbers as confirmed results.

Ties go to the simpler model.

## 6. Where to aim

The failing rows depend on which base model you improve — a GNN contributor
fails CG hardest while the categorical linear fails RCA, and the multinomial
punisher fails only PD while gaussian/ridge fail everything *but* PD. So do
not work from a fixed target list: fetch your own model's deficit profile,
then declare targets from it.

**Where the numbers live:**

| resource | what it gives you |
|---|---|
| `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv` | every score: 32 stacks x 21 rows |
| `.../23_stack_sweep_updated/slot_report.jpg` | each slot option's rows, averaged over the other two slots |
| `.../23_stack_sweep_updated/slot_concordance.jpg` | whether a deficit / ranking is stable across contexts (Kendall's W) |
| `plots/simulation/23_*/evaluation/scores.csv` + `visuals/` | per-stack scores and one figure per row |
| `notes/evaluation_metric_defs.md` | what each row measures |
| PRs #140 and #143 | the narrative: findings, shortcomings, per-slot verdicts |

**How to read it:** filter `score_matrix.csv` to the contexts containing your
base model, average over the other two slots, and rank your slot's rows with
score >= 2 — that is your target list. Check `slot_concordance` before
committing to a target: a deficit that appears in only one context is noise,
not a direction.

**Known context that constrains solutions, whatever the base model:** the
group-spread rows (CG, PD) and SC share one root cause — independent
per-agent sampling ignores between-participant correlation (the motivating
review comment on PR #140); every current model sits on the independence
floor there. CG is *anti-correlated* with the individual-fit rows
(r ~ -0.7 to -0.9), so buying group spread by making individual behavior
worse is the known failure mode. The switching deficit is confined to the
first decision round (founding exodus, human mean net flow 2.42 vs sim ~1.5)
— rates (SA/SB) and post-exodus stickiness are already matched.

## 7. Tooling

All config-driven; the standard simulation protocol is the 23-family
template (2 groups x 8 agents, 24 rounds, 100 episodes, seed 42,
`save_per_round: true`).

| step | command | where |
|---|---|---|
| train GNN | `scripts/train_cluster.sh ah <config>` | Raven |
| train linear | `scripts/baselines/` runners | local |
| simulate | `scripts/simulate_cluster.sh <config>` | Raven |
| fetch results | `scripts/fetch_cluster.sh <remote_path>` | local |
| evaluate | `python -m aimanager evaluate <sim config>` | local |
| sweep (Stage 2) | `python scripts/data_analysis/evaluation_sweep.py <name> <sim dirs>` | local |
| tests | `scripts/remote_test.sh` (PyG) / `pytest` (eval suite) | Raven / local |

Work happens on a dedicated branch named `auto/<slot>-<slug>` (e.g.
`auto/contribution-group-context-feature`), checked out in its **own git
worktree** — agents run in parallel and never share a checkout. Name new
configs, artifacts, and sim output dirs with the same slug so parallel runs
cannot collide on paths either. Commits via the `/commit` skill; one
experiment = one branch = one PR.

## 8. Frozen surface

Never modified by agents, under any experiment:

- `src/aimanager/evaluation_suite/` (all of it),
- `notes/evaluation_metric_defs.md` and `notes/eval_scoring_schema.md`,
- `experiments/` (the human data),
- scoring parameters (500 repeats, master seed 42) and the simulation
  protocol (episode count, seeds, game parameters),
- the reference stack definition and other branches' (or merged) log files.

If an experiment seems to require touching any of these, stop and escalate
to the human maintainer.

## 9. The loop

1. Read this file, the merged logs in `notes/autoresearch_log/`, and your
   base model's deficit profile (§6).
2. Create your branch and worktree (§7); declare the experiment in your log
   file: slot, base model, target row(s), hypothesis, planned change.
3. Implement the one change; train; run Stage 1.
4. Log the result (§10) — improvements and failures alike.
5. Kept per §2 (targets improved, nothing regressed)? Run Stage 2.
   Confirmed? Open a PR with both stages' numbers. Otherwise revert, log,
   pick the next hypothesis.
6. Repeat.

## 10. Results log

One log file per experiment branch, so parallel merges never clash:
`notes/autoresearch_log/<branch-slug>.md` — each branch touches only its own
file. Three sections, in this order:

1. **Declaration** — slot, base model, target rows, hypothesis, planned
   change.
2. **Results** — one row per run:

   | date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
   |---|---|---|---|---|---|---|

3. **Notes** — a numbered list (`1.`, `2.`, `3.`, ...), appended as you go,
   recording the path you actually took: what you observed, what you decided
   and why, dead ends and what killed them. Sparse but informative — one to
   three sentences per entry, only at real decision points, so a reader can
   reconstruct where the model went without reading the diffs.

Scores are reported exactly as computed — no rounding a 1.04 down, no
re-running for a better draw. A failed experiment logged well is a
contribution; a gamed number poisons every comparison after it. Merged log
files are never edited afterwards.
