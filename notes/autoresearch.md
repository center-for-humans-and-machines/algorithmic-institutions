# Autoresearch: Optimizing the Artificial-Human Stack

A standing guideline for research agents that improve the artificial-human
models (contribution / switch / punisher) against the evaluation suite. The
specific campaign an agent runs lives in its own document; this file defines
the objective, the rules, and the loop.

---

## 1. Mission

Make the simulated stack indistinguishable from the human games. Progress is
measured by the evaluation suite only (`python -m aimanager evaluate`, row
definitions in `notes/evaluation_metric_defs.md`).

## 2. The metrics

Three numbers, judged in this order, all from one `evaluation/scores.csv`
(21 rows):

1. **Target rows** — the rows your hypothesis declares (candidates from §6;
   e.g. CG, RCA, RCD for the GNN contributor), collectively or a selection.
   Their scores must drop; no movement means the hypothesis failed, whatever
   else moved.
2. **Rows <= 1** — rows at or below the human-vs-human noise ceiling, whole
   stack. Must not fall; raising it is the headline. **Baseline: 11 / 21.**
3. **Mean score** — average over all 21 rows. Must not rise; breaks ties.
   **Baseline: 1.76.**

An experiment is **kept** iff its target rows improve and neither stack
metric regresses — you cannot buy your targets by quietly breaking the rest
of the stack.

**Success additionally requires a band upgrade.** The scoring bands
(<= 1 / 1-2 / 2-5 / > 5) are the classes: at least one declared target row
must finish in a better band than it started — from > 5 into 2-5, from 2-5
into 1-2 or <= 1, from 1-2 into <= 1 — in the reference stack, confirmed by
Stage 2. A within-band improvement, however large, is a `[FAIL]` with
valuable notes, not a success.

## 3. Evaluation protocol (two-stage)

The metrics are a property of a full stack, so candidates are always scored
inside one. **Reference stack** (current) — only the human maintainer
updates this definition, when a candidate is accepted:

| slot | model | artifact |
|---|---|---|
| contribution | `gnn` | `artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| switch | `gnn` | `artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` |
| punisher | `lin_multinomial` | `artifacts/baselines/punishment_multinomial_best_with_contr.joblib` |

Reference sim config (also carries the shared `valid_model`, which is
plumbing, not a slot):
`configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch.yml`.

- **Stage 1 — iterate.** Swap your candidate into its slot of the reference
  stack; one simulation (§7 protocol) + one evaluation.
- **Stage 2 — confirm.** The full sweep: the 8-config family with your
  candidate replacing its slot everywhere, then `evaluation_sweep.py`. The
  claim stands only if the candidate also wins its slot across contexts
  (the Kendall's-W discipline of PR #143), not just in the reference stack.

## 4. Agents and slots

One slot per agent: **contribution**, **switch**, or **punisher**. You change
only your slot's model, features, and training configs — one change per
experiment, declared in your log file before you start (§10).

A bug fix in shared code (encoder, simulation, preprocessing) is legal but is
its own experiment: fix only, before/after scores for the reference stack.

## 5. Legal and illegal changes

**Legal** — anything that plausibly makes the model more *human*, built on a
direction the evaluations point to (§6) or a finding you make and document:

- architecture changes,
- new input features — only information the real player or manager observably
  had at decision time (punishment models condition on round t-1, never on
  the current round's contributions),
- hyperparameter search, including selecting between variants by Stage-1
  score,
- training-data handling within the conventions (GNNs train on the
  flip-doubled data, linears on the single copy),
- bug fixes, with an explanation of what was wrong.

Every change carries a one-sentence behavioral rationale: *which* human
behavior it captures and *which* row that should move. If you cannot write
that sentence, the change is a frankenstein — do not make it.

**Illegal** — anything that improves the number without improving the model:

- touching the frozen surface (§8),
- changing or shopping simulation / scoring seeds and episode counts,
- features engineered at a metric's definition rather than at behavior
  (e.g. keyed to a bin edge or stratum boundary),
- training on the evaluation's resampling structure, or on the flipped
  duplicates where the convention says single-copy,
- reporting Stage-1 numbers as confirmed results.

Ties go to the simpler model.

## 6. Where to aim

The failing rows depend on the base model — the GNN contributor fails CG
hardest, the categorical linear fails RCA; the multinomial punisher fails
only PD, gaussian/ridge everything *but* PD. Do not work from a fixed
target list: fetch your base model's deficit profile, then declare targets.

**Where the numbers live:**

| resource | what it gives you |
|---|---|
| `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv` | every score: 32 stacks x 21 rows |
| `.../23_stack_sweep_updated/slot_report.jpg` | each slot option's rows, averaged over the other slots |
| `.../23_stack_sweep_updated/slot_concordance.jpg` | whether a deficit / ranking is stable across contexts |
| `plots/simulation/23_*/evaluation/scores.csv` + `visuals/` | per-stack scores and one figure per row |
| `notes/evaluation_metric_defs.md` | what each row measures |
| PRs #140 and #143 | the narrative: findings, shortcomings, per-slot verdicts |

**How to read it:** filter `score_matrix.csv` to your base model's contexts,
average over the other two slots, rank your slot's rows with score >= 2 —
that is your target list. Check concordance first: a deficit that appears in
one context is noise, not a direction.

**Known constraints, whatever the base model:** CG, PD, and SC share one
root cause — independent per-agent sampling ignores between-participant
correlation (the motivating comment on PR #140); every current model sits on
the independence floor there. CG is *anti-correlated* with the
individual-fit rows (r ~ -0.7 to -0.9): buying group spread with worse
individual behavior is the known failure mode. The switching deficit is
confined to the first decision round (founding exodus, human mean net flow
2.42 vs sim ~1.5) — rates (SA/SB) and post-exodus stickiness already match.

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

## 9. Work process

**Roles.** The orchestrator is a **Fable** model; planning, implementation,
and analysis are delegated to **Opus** subagents, one task at a time. The
orchestrator decides, validates, and confirms; subagents execute — they
never decide scope.

**Branch and worktree.** Every experiment lives on its own branch,
`auto/<slot>-<slug>`, checked out in its **own git worktree** — parallel
experiments never share a checkout. Name new configs, artifacts, and sim
output dirs with the same slug so runs cannot collide on paths either.
Commits via the `/commit` skill; one experiment = one branch = one PR.

**Commit identity.** Autoresearch commits are authored by Claude, not the
human — the human only reviews and merges. In the experiment worktree, set
before the first commit:

```bash
git config extensions.worktreeConfig true
git config --worktree user.name "Claude"
git config --worktree user.email "noreply@anthropic.com"
```

With Claude as the author, no `Co-Authored-By` trailer is added. Pushes and
PRs still go through the human's account (transport only); a dedicated
machine account may replace this later.

**The loop:**

1. Read this file, the merged logs in `notes/autoresearch_log/`, the
   `[FAIL]` PRs of prior experiments, and your base model's deficit
   profile (§6).
2. Create the branch and worktree; write the declaration in your log
   file (§10).
3. **Plan** — a planning subagent turns the hypothesis into a **simple
   numbered list of steps**, nothing more elaborate. The orchestrator
   validates it before anything runs — targets per §2, every step legal per
   §5, nothing on the frozen surface (§8) — then records it in the log file
   and commits it.
4. **Implement** — one subagent per step. The orchestrator confirms each
   step's result before dispatching the next and commits at each confirmed
   step — commits map to steps, never one monolith. If a step reveals the
   plan is wrong, revise the step list first (through validation again),
   then continue.
5. Train, simulate, evaluate per §3 and §7; log every run (§10).
6. Kept per §2 **with a band upgrade** on a target row? Run Stage 2. Kept
   but within-band? Skip the sweep — it cannot become a success.
7. **Every experiment ends in a PR** — titled `[SUCCESS] ...` (band upgrade,
   Stage-2 confirmed) or `[FAIL] ...` (targets did not move, no band
   upgrade, or Stage 2 did not confirm; never merged — it exists so the
   next agent does not retry it).
   No silent abandonment. The body, in order:
   1. **Hypothesis** — brief: the behavioral claim, the planned change, and
      the targeted rows with their starting scores.
   2. **Results** — the log file's results table (§10), both stages where
      run.
   3. **Collateral** — non-target rows that moved, grouped `+` / `-`. Only
      the important ones: movements that could seed further experiments,
      not every wiggle.
8. Next hypothesis = new experiment: new branch, new worktree, new PR.

## 10. Results log

One log file per experiment branch, so parallel merges never clash:
`notes/autoresearch_log/<branch-slug>.md` — each branch touches only its own
file. Four sections, in this order:

1. **Declaration** — slot, base model, target rows, hypothesis, planned
   change.
2. **Plan** — the validated step list from §9, checked off as steps are
   confirmed.
3. **Results** — one row per run:

   | date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
   |---|---|---|---|---|---|---|

4. **Notes** — a numbered list (`1.`, `2.`, ...), appended as you go: what
   you observed, what you decided and why, dead ends and what killed them.
   One to three sentences per entry, only at real decision points, so a
   reader can reconstruct where the model went without reading the diffs.

Scores are reported exactly as computed — no rounding a 1.04 down, no
re-running for a better draw. A failed experiment logged well is a
contribution; a gamed number poisons every comparison after it. Merged log
files are never edited afterwards.
