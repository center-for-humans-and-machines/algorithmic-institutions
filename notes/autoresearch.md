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

Everything comes from one `evaluation/scores.csv` (22 rows), judged against
the evaluation stack's own baseline scores (§3; on a parent `[SUCCESS]` PR,
the parent's — §9). One protected row, then two gates, all required for
success:

**Protected row: RCE** (punishment response slope, the OLS slope of the
next-round contribution change on the punishment received, per contribution
band 0-4 / 5-9 / 10-14 / 15-19; humans +0.14 / +0.10 / -0.08 / -0.16). An
experiment may not band-downgrade RCE against its baseline, may not flip
any of the four band slopes away from the human sign, and may not halve any
band's slope magnitude. Any of the three is a `[FAIL]`, whatever the gates
below say.

The magnitude clause carries two qualifications, both added after it
misfired twice on its first outing (PRs #190 and #192). A band's |slope|
falling to half its baseline value or below is a `[FAIL]` **only if** (a)
the candidate's slope is not closer to the human value than the baseline's
was, and (b) the change exceeds one pooled standard error of the two
slopes. Qualification (a) exists because the clause is a magnitude test on
a signed quantity: on the 10-14 band of PR #192's reference stack it fired
on a slope moving from +0.013, the wrong sign, to -0.001, toward the human
-0.077 — an improvement. Qualification (b) exists because the 10-14 band's
baseline magnitude is routinely around 0.05, so a single-seed fluctuation
of 0.03 trips a relative threshold; that band is also the one PR #183
showed is never learned, with the wrong sign teacher-forced and held out.
Report every band's slope with its standard error, its row count, and the
change in pooled standard errors, so a firing can be read. Neither
qualification changes a verdict recorded before it: PRs #190 and #192 both
failed gate 1 independently.

**All three clauses are subject to the symmetry rule.** RCE's own seed sd
is 0.106 and it is one of the ten ungateable rows — it sits in band <= 1 in
exactly one arm of six — so its band-drop clause fires only on a drop
larger than 0.106, and the sign clause is **retired on the 10-14 and 15-19
bands**, where retraining an unchanged model flips the sign on its own. The
0-4 and 5-9 bands keep the sign clause; they are stable across the six
arms. Punishment-response experiments — punisher-slot changes, and any
experiment whose declared target is the contributor's reaction to
punishment — are judged on RCE: it is their target row for gate 1.

1. **A band upgrade on a target row.** The scoring bands
   (<= 1 / 1-2 / 2-5 / > 5) are the classes: at least one row your
   hypothesis declares (candidates from §6) must finish in a better band
   than its baseline — from > 5 into 2-5, from 2-5 into 1-2 or <= 1, from
   1-2 into <= 1. A within-band improvement, however large, is a `[FAIL]`
   with valuable notes, not a success. **The upgrade must also clear the
   noise floor**: the target row's move must exceed that row's seed
   standard deviation (table below). A band crossed by less than the floor
   is a lucky draw, not a result, and the ten rows marked ungateable below
   cannot serve as a gate-1 target on a single run at all.
2. **The mean score holds.** The average over all 22 rows may not rise
   more than 10% above the evaluation stack's baseline mean (e.g. baseline
   1.76 -> ceiling 1.936). A band upgrade is allowed to cost a little
   elsewhere — but not to be bought by breaking the rest of the stack.
   The mean's own seed standard deviation is 0.047 against a margin of
   about 0.103 on the current frontier, so this gate already sits at
   roughly two floors and is left as it is. It is the backstop that the
   symmetry rule below deliberately relies on: many small real losses, each
   individually under its row's floor, still show up here.

**The noise floor, and the symmetry rule.** PR #195 ran the same
contributor architecture six ways — five seeds plus the shipped artifact —
through one stack with everything else held identical, so only the training
draw varied. The spread is larger than most experiments move:

| row | seed sd | gateable on one run |
|---|---|---|
| RPA | 0.018 | yes |
| PB | 0.023 | yes |
| RPB | 0.028 | yes |
| PC | 0.036 | yes |
| PA | 0.040 | yes |
| SB | 0.046 | **no** |
| CE | 0.058 | yes |
| PD | 0.059 | yes |
| RCE | 0.106 | **no** |
| CC | 0.133 | **no** |
| SC | 0.136 | yes |
| RCA | 0.141 | yes |
| CF | 0.141 | **no** |
| RCB | 0.142 | yes |
| SA | 0.162 | **no** |
| RCC | 0.163 | yes |
| RSA | 0.155 | **no** |
| CA / CD | 0.188 | **no** |
| CB | 0.215 | **no** |
| RCD | 0.270 | yes (marginal) |
| CG | 0.301 | **no** |
| 22-row mean | 0.047 | |
| rows <= 1 | 3.16 (range 6 to 14) | |

A row is **ungateable on one run** when a band boundary falls inside one
seed sd and the six arms genuinely land in two bands. Ten do: CA, CB, CC,
CD, CF, CG, SA, SB, RCE and RSA. **Rows <= 1 is the worst of all** and must
not be quoted as a property of a model: six rows are always at the ceiling,
six never are, and ten flip on the training draw alone.

**The symmetry rule: no movement smaller than its row's seed sd counts
either for or against an experiment.** A target must clear the floor to
earn a gate-1 upgrade, and equally, a row worsening by less than its floor
is not a cost — it is a retrain. Applying the threshold to gains but not to
losses would make improvement nearly impossible, since one row would have
to beat the noise to help while twenty-one could hurt by luck. Gate 2 is
the backstop against many sub-floor losses accumulating. Report every
movement with its seed sd beside it, and mark those under the floor as not
distinguishable from a retrain.

**The frontier baseline is the six-arm mean, not the shipped run.** PR #195
also found that the shipped contributor ranks first of six on every
headline aggregate — best mean, most rows at the ceiling, best on 9 of 22
rows, five times closer to the human contribution level than any member —
while PR #188 showed its training fit is an ordinary draw. The plausible
reading is selection: it became the frontier by scoring well on this
evaluation, so candidates since have been measured against a favourable
tail. The frontier stack's baseline is therefore the **six-arm mean score
vector**, `plots/data_analysis/evaluation/seed_spread_noise_floor/per_row.csv`
on `auto/seed-spread-noise-floor`, headline figures:

| | shipped run (old) | six-arm mean (new) |
|---|---|---|
| 22-row mean | 1.0331 | **1.0923** |
| rows <= 1 | 14 | **10** |
| RCC | 1.2969 | **1.5734** |
| RCE | 0.8823 | **1.0683** |

Three consequences, all deliberate. This is a change to the **scoreboard,
not to the model**: an average cannot be simulated, so the artifact
candidates are built from is unchanged; only the numbers they are judged
against move. Scores recorded against the shipped run before this — PRs
#193, #194 and #196 — are not comparable with scores after it, though
their conclusions mostly stand. And the reset applies **only to the
frontier stack**, the one where the six arms were actually run; every other
evaluation stack keeps its single-run baseline until the same measurement
is made there, and a candidate on such a stack should say so.

Nothing else gates. **Rows <= 1** (rows at or below the human-vs-human
noise ceiling) is still computed and reported in every results table (§10),
in the same column as always — context for the reader, not a criterion.

**Re-baseline (`auto/punisher-current-contribution`).** Until that branch,
both artificial punishers conditioned on round t-1's contribution while the
human manager punishes round t's
(`notes/autoresearch_log/punisher-current-contribution.md`). The fix
retrains both punisher families, and since one punisher artifact sits in
every stack it moves every row of every stack. The ledger's baselines — the
score matrix and ranking of §3 and §6, and the confirmed scores of the
frontier PRs — are therefore reset by that branch's stage D; scores
recorded before it (21 rows, lagged punisher) are not comparable with scores
after it (22 rows, RCE included, current-contribution punisher). Stage D
re-ran the top stack and the four frontier stacks; the post-fix baselines
are the table in §3.

**Frozen noise model, and how contributor-trunk changes are judged.** The
copula parameters -- the correlation strength rho and the persistence phi,
per model family -- are frozen (§8). No experiment recalibrates them as a
side effect of changing a trunk; altering either is its own declared
experiment. The reason is attribution: a recalibration riding along with a
trunk change makes the two indistinguishable.

For the same reason, a **contributor-trunk change is judged with the copula
disabled**, on the state-spread diagnostic, alongside the usual gates. The
diagnostic is the decomposition Var(c) = Var(E[c | history]) +
Var(residual) over a free-running simulation, against the human histories
(`scripts/data_analysis/copula_closed_loop_variance.py`, §7). Human
Var(E[c|hist]) is 27.9; the stimulus-skip trunk with the copula disabled
sits at 18.9 with its residual variance already correct at 11.4. That gap
is the open defect the group-spread row CG only indirectly reports: the
copula's episode-long persistence supplies most of CG by compounding (a
factor of 5.3 on the group's episode mean), and by the late rounds the
carried state has absorbed the latent, so CG measured with the copula on
does not tell you whether a trunk change helped. Report the copula-off
Var(E[c|hist]) for any contributor-slot candidate.

Evidence for both rules: `notes/autoresearch_log/copula-closed-loop-variance.md`,
`copula-missing-state.md` and `copula-seed-ensemble.md`. The human residual
dependence is a round-local shock with a one-round echo, not the
episode-long latent that is shipped, but the shipped shape is kept for now
because nothing yet replaces the variance it supplies; see
`doc/plans/post-rebaseline-program.md`.

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
evaluates inside `gnn x lin x multinomial` (pre-fix sweep figures: rows
<= 1: 9/21, mean 1.845 — that stack has no post-fix sim yet; re-run it with
the current-contribution punisher before using it as a baseline); GNN
contribution, GNN switch, and multinomial punisher candidates all evaluate
inside the top stack itself.

Artifact paths for any stack are read off its sim config,
`configs/simulation/manager_testing/23_2g8a_self_<contr>_contr_<switch>_switch.yml`
(which also carries the shared `valid_model` — plumbing, not a slot). The
current top of the ranking, `gnn x gnn x multinomial` (post-fix, 22 rows:
rows <= 1: 13/22, mean 1.7405; sim
`plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_curpun`, run
`lin_multinomial_self`; the pre-fix sweep figure was 11/21, mean 1.759):

| slot | model | artifact |
|---|---|---|
| contribution | `gnn` | `artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` |
| switch | `gnn` | `artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt` |
| punisher | `lin_multinomial` | `artifacts/baselines/punishment_multinomial_current_contr.joblib` (copula-stamped copy for the frontier stacks: `punishment_multinomial_current_contr_severity_copula.joblib`) |

**Post-fix baselines (stage D of `auto/punisher-current-contribution`,
22 rows, RCE included; full tables in
`plots/data_analysis/evaluation/punisher_current_contr/rebaseline_table.md`).**
These replace the confirmed scores in the frontier PRs' bodies and the
pre-fix rows of the score matrix for these stacks; a successor of one of
these stacks is judged against the row here, at full precision in the
`_curpun` sim's `evaluation/scores.csv`:

| stack (sim dir `plots/simulation/<...>_curpun`) | punisher | rows <= 1 | mean | RCE (bands 0-4 / 5-9 / 10-14 / 15-19) |
|---|---|---|---|---|
| PR #179 `23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch` | lin_multinomial copula | 9/22 | 1.1079 | 1.2682 (+0.049 / -0.007 / -0.017 / +0.040) |
| PR #181 `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` | lin_multinomial copula | 13/22 | 1.0357 | 0.8942 (+0.095 / +0.020 / -0.058 / -0.160) |
| PR #177 `23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch` | lin_multinomial copula | 12/22 | 1.1012 | 0.8508 (+0.068 / +0.109 / -0.020 / -0.193) |
| PR #174 `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch` | lin_multinomial copula | 8/22 | 1.1880 | 0.7046 (+0.120 / +0.071 / -0.112 / -0.205) |
| main `23_2g8a_self_gnn_contr_gnn_switch` | lin_multinomial | 13/22 | 1.7405 | 0.9976 (+0.064 / +0.092 / +0.044 / -0.098) |
| main `23_2g8a_self_gnn_contr_gnn_switch` | gnn | 8/22 | 1.7094 | 1.0048 (+0.073 / +0.050 / +0.013 / +0.029) |

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
  had at decision time (contribution models condition on round t-1;
  punishment models may condition on the current round's contributions —
  the manager sees them before punishing — but never on the current round's
  punishments, payoffs or common good),
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
| `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv` | every score: 32 stacks x 21 rows (pre-fix sweep, lagged punisher — deficit profiles only; post-fix baselines for the six re-run stacks, 22 rows, are in §3 and `plots/data_analysis/evaluation/punisher_current_contr/rebaseline_table.csv`) |
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
| state-spread diagnostic (contributor slot) | `python scripts/data_analysis/copula_closed_loop_variance.py` | local |
| tests | `scripts/remote_test.sh` (PyG) / `pytest` (eval suite) | Raven / local |

## 8. Frozen surface

Never modified by agents, under any experiment:

- `src/aimanager/evaluation_suite/` (all of it),
- `notes/evaluation_metric_defs.md` and `notes/eval_scoring_schema.md`,
- `experiments/` (the human data),
- scoring parameters (500 repeats, master seed 42) and the simulation
  protocol (episode count, seeds, game parameters),
- the copula parameters per model family (rho and phi): contribution
  rho 0.0395 / phi 1.0, punisher severity rho 0.4273, switch rho 0.1165 /
  phi 0.704 -- a retrain of the model a copula is stamped on carries the
  frozen value over rather than refitting it (§2),
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
   upgrade on a target row with the mean inside the 10% margin is a
   success; anything less is a fail. There is no second stage.
7. **Every experiment ends in a PR** — titled `[SUCCESS] ...` (band upgrade
   on a target row, mean within the margin) or `[FAIL] ...` (no band
   upgrade, or the mean rose past it; never merged — it exists so the next
   agent does not retry it).
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
