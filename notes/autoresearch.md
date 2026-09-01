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

Everything comes from one `evaluation/scores.csv` (21 rows), judged against
the evaluation stack's own baseline scores (§3; on a parent `[SUCCESS]` PR,
the parent's — §9). Two gates, both required for success:

1. **A band upgrade on a target row.** The scoring bands
   (<= 1 / 1-2 / 2-5 / > 5) are the classes: at least one row your
   hypothesis declares (candidates from §6) must finish in a better band
   than its baseline — from > 5 into 2-5, from 2-5 into 1-2 or <= 1, from
   1-2 into <= 1. A within-band improvement, however large, is a `[FAIL]`
   with valuable notes, not a success.
2. **The mean score improves.** The average over all 21 rows must drop
   below the evaluation stack's baseline mean — you cannot buy your targets
   by breaking the rest of the stack.

Nothing else gates. **Rows <= 1** (rows at or below the human-vs-human
noise ceiling) is still computed and reported in every results table (§10),
in the same column as always — context for the reader, not a criterion.

## 3. Evaluation protocol

The metrics are a property of a full stack, so candidates are always scored
inside one — the **highest-ranked stack that contains your base model**:
rank the sweep's stacks (`score_matrix.csv`, §6) by rows <= 1, descending,
ties broken by the lower mean score; filter to those with your base model
in your slot, take the best. Swap your candidate into its slot there; one
simulation (§7 protocol) + one evaluation. That stack's own scores are the
baseline for both gates (§2): its row in `score_matrix.csv`, or at full
precision
`plots/simulation/23_2g8a_self_<contr>_contr_<switch>_switch/evaluation/scores.csv`
with the `run` column filtered to your punisher pairing. There is no
confirmation sweep — winning in your base model's best context is the
claim. (When the maintainer targets a parent `[SUCCESS]` PR, the stack and
baseline come from the parent instead — §9.) E.g. a lin-switch candidate
evaluates inside `gnn x lin x multinomial` (rows <= 1: 9/21, mean 1.845);
GNN contribution, GNN switch, and multinomial punisher candidates all
evaluate inside the top stack itself.

Artifact paths for any stack are read off its sim config,
`configs/simulation/manager_testing/23_2g8a_self_<contr>_contr_<switch>_switch.yml`
(which also carries the shared `valid_model` — plumbing, not a slot). The
current top of the ranking, `gnn x gnn x multinomial` (rows <= 1: 11/21,
mean 1.759):

| slot | model | artifact |
|---|---|---|
| contribution | `gnn` | `artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| switch | `gnn` | `artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` |
| punisher | `lin_multinomial` | `artifacts/baselines/punishment_multinomial_best_with_contr.joblib` |

Only the human maintainer refreshes the score matrix (and with it this
ranking), when a candidate is accepted.

## 4. Agents and slots

One slot per agent: **contribution**, **switch**, or **punisher**. You change
only your slot's model, features, and training configs — one change per
experiment, declared in your log file before you start (§10).

A bug fix in shared code (encoder, simulation, preprocessing) is legal but is
its own experiment: fix only, before/after scores for the top-ranked stack
(§3).

## 5. Legal and illegal changes

**Legal** — anything that plausibly makes the model more *human*, built on a
direction the evaluations point to (§6) or a finding you make and document:

- architecture changes,
- new input features — only information the real player or manager observably
  had at decision time (punishment models condition on round t-1, never on
  the current round's contributions),
- hyperparameter search, including selecting between variants by their
  evaluation score,
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
- stack-shopping: evaluating in any stack other than the one §3 selects
  for your base model, or reporting scores from a friendlier context.

Ties go to the simpler model.

**Iteration budget.** The loop lives on fast retrains, so wall-clock is a
constraint, not a footnote. Before adopting a method, check your base
model's current training time from recent plain runs (the SLURM logs of
the latest `train-ah` jobs on Raven); any method that needs **more than
3x that** is ruled out, whatever it promises. Example: scheduled
sampling — it replaces teacher-forced parallel batches with sequential
own-rollout unrolling, pushing one training to ~1.5 h; at several
variants per hypothesis that turns a same-day experiment into a
multi-day one, which is why the schedsamp family is vetoed (PR #163).

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
one context is noise, not a direction. (Building on a parent `[SUCCESS]`
PR: the matrix does not contain the parent's candidate — the deficit
profile comes from the parent's own `evaluation/scores.csv` instead, §9.)

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
| sweep (maintainer matrix refresh) | `python scripts/data_analysis/evaluation_sweep.py <name> <sim dirs>` | local |
| tests | `scripts/remote_test.sh` (PyG) / `pytest` (eval suite) | Raven / local |

## 8. Frozen surface

Never modified by agents, under any experiment:

- `src/aimanager/evaluation_suite/` (all of it),
- `notes/evaluation_metric_defs.md` and `notes/eval_scoring_schema.md`,
- `experiments/` (the human data),
- scoring parameters (500 repeats, master seed 42) and the simulation
  protocol (episode count, seeds, game parameters),
- the evaluation-stack selection (§3) — the sweep's score matrix and the
  ranking rule — and other branches' (or merged) log files.

If an experiment seems to require touching any of these, stop and escalate
to the human maintainer.

## 9. Work process

**Roles.** **Fable** opens the experiment and nothing more: research,
declaration, plan (loop steps 1-3). **Opus**
orchestrates everything after — on receiving the declaration and the plan
it validates the plan and attaches an implementer to every step: an
**Opus** engineer where the step is complicated or risky, a **Sonnet**
agent otherwise, for cost efficiency. It then dispatches subagents one
step at a time, each to the model its step carries, and confirms each
result. Subagents execute; they never decide scope.

**Branch and worktree.** Every experiment lives on its own branch,
`auto/<slot>-<slug>`, checked out in its **own git worktree** under
`.claude/worktrees/<slug>` — parallel experiments never share a checkout. Name new configs, artifacts, and sim
output dirs with the same slug so runs cannot collide on paths either.
Commits via the `/commit` skill; one experiment = one branch = one PR.
Branches start from `main` — unless the maintainer targets a parent PR:

**Building on a `[SUCCESS]` PR.** Successful PRs are not always merged;
their branches stay alive as the frontier. When the maintainer points your
experiment at one, base yourself on it instead of `main`: fetch the PR's
head branch and create your branch and worktree from it
(`git worktree add <dir> -b auto/<slot>-<slug> origin/<parent-branch>`),
and open your PR with `--base <parent-branch>` so the diff shows only your
own change. Read the parent's log file before planning. Your evaluation
runs in the same stack the parent's did, with the parent's candidate as
your base model — the parent's confirmed scores are the baseline the §2
gates are judged against: the results table in the parent's PR body and
log file, backed by the parent branch's own sim output
(`plots/simulation/<parent-slug...>/evaluation/scores.csv`, in your
worktree since you branched off it). Name the parent PR in your
declaration (§10).

**Remote isolation.** The same rule holds on Raven: parallel experiments
never share the remote checkout. Every `train_cluster.sh` /
`simulate_cluster.sh` / `fetch_cluster.sh` call from an experiment worktree
sets `AI_REMOTE_DIR='~/autoresearch/<slug>'`, which syncs and runs in that
dir instead of the shared `~/algorithmic-institutions` — the shared
checkout is synced from `main` only and owns the single venv. Isolated
dirs carry no venv: the scripts wire the shared venv plus the dir's own
`src/` via PYTHONPATH into the jobs automatically, so each job imports
exactly its branch's code. Outputs land inside the dir; fetch from there.
When the experiment's PR closes, delete the remote dir.

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
3. **Plan** — Fable turns the hypothesis into a numbered list of
   implementation steps in the style of
   [signifier-trainer#13](https://github.com/cemrtkn/signifier-trainer/issues/13):
   clearly separated steps that build on each other, each opening with a
   bold name and the exact place to change (file, function, config; new or
   existing), then what exactly changes there — understandable and
   concise. The orchestrator validates it before anything runs — targets
   per §2, every step legal per §5, nothing on the frozen surface (§8) —
   attaches an implementer to every step (Opus or Sonnet, per Roles), then
   records the tagged plan in the log file and commits it.
4. **Implement** — one subagent per step, sent to the model its step
   carries. The orchestrator confirms each step's result
   before dispatching the next and commits at each confirmed step —
   commits map to steps, never one monolith. If a step reveals the plan is
   wrong, revise the step list first (through validation again), then
   continue.
5. Train, simulate, evaluate per §3 and §7; log every run (§10).
6. The verdict comes straight from that single evaluation, per §2: a band
   upgrade on a target row *and* a better mean is a success; anything less
   is a fail. There is no second stage.
7. **Every experiment ends in a PR** — titled `[SUCCESS] ...` (band upgrade
   on a target row and the mean improved) or `[FAIL] ...` (no band upgrade,
   or the mean did not improve; never merged — it exists so the next agent
   does not retry it).
   No silent abandonment. The body, in order:
   1. **Hypothesis** — brief: the behavioral claim, the planned change, and
      the targeted rows with their starting scores.
   2. **Results** — the log file's results table (§10).
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
2. **Plan** — the validated, implementer-tagged step list from §9, checked
   off as steps are confirmed.
3. **Results** — one row per run:

   | date | change (one line) | target scores | rows <= 1 | mean | verdict |
   |---|---|---|---|---|---|

4. **Notes** — a numbered list (`1.`, `2.`, ...), appended as you go: what
   you observed, what you decided and why, dead ends and what killed them.
   One to three sentences per entry, only at real decision points, so a
   reader can reconstruct where the model went without reading the diffs.

Scores are reported exactly as computed — no rounding a 1.04 down, no
re-running for a better draw. A failed experiment logged well is a
contribution; a gamed number poisons every comparison after it. Merged log
files are never edited afterwards.
