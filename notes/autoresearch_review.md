# Autoresearch Review: Auditing the Two Frontier Lineages

A standing guideline for the code review of the autoresearch campaign
(`notes/autoresearch.md`). The summary (`notes/autoresearch_summary.md`)
reports *what the campaign claimed*; this review asks *whether the code
behind those claims deserves to be adopted*. It reads the two frontier
lineages beginning to end and produces one findings document. It changes
no experiment branch, re-runs no experiment, and re-judges no verdict.

---

## 1. What this review is for

Nine `[SUCCESS]` PRs stand on two unmerged lineages. Whatever is adopted
into `main` comes from them, so every feature they added has to survive
four questions:

- **Is it duplicated?** Two implementations of one mechanism, or a step
  that a later step made redundant.
- **Is it wrong, or is it cheating?** A bug that changes simulated
  behavior, or a number bought by something other than a better model.
- **Is it spoon-fed?** A gain handed to the model by the analyst —
  hand-stamped constants, parameters tuned against the evaluation,
  features keyed to a metric's definition rather than to behavior. §5 of
  `notes/autoresearch.md` draws the legal line; the interesting findings
  sit just on either side of it.
- **Is it convoluted, or does it fit?** The right mechanism in the wrong
  layer, a change carried by more scaffolding than substance, a feature
  with exactly one consumer.

The output is not a patch. It is a ranked list of findings that tells the
maintainer what to adopt as-is, what to adopt after a fix, and what to
leave on the branch.

## 2. Scope

The two lineages, root first. Each step is reviewed against **its own
parent**, which is also how it was judged.

**gnn lineage** — `main -> #160 -> #165 -> #171 -> #179`, tip mean 1.099

| order | PR | branch | slot | primary surface |
|---|---|---|---|---|
| 1 | #160 | `auto/punisher-severity-copula-v2` | punisher | `simulation/linear_ah.py`, `punishment_copula_rho.py` |
| 2 | #165 | `auto/contribution-herding-copula-v2` | contribution | `generic/copula.py`, `generic/graph.py`, `contribution_copula_rho.py`, `make_contribution_copula_artifact.py` |
| 3 | #171 | `auto/switch-joint-exodus` | switch | `generic/joint_exodus.py`, `generic/conditional_bernoulli.py`, `artificial_humans/train.py`, `generic/graph.py`, 7 cluster scripts |
| 4 | #179 | `auto/contribution-group-vnode` | contribution | `generic/group_vnode.py`, `generic/graph.py`, 3 sim configs, 3 slurm stampers |

**gmlp lineage** — `main -> #167 -> #170 -> #172 -> #174 -> #177`, tip
mean 1.1249

| order | PR | branch | slot | primary surface |
|---|---|---|---|---|
| 1 | #167 | `auto/contribution-gaussian-mlp-v2` | contribution | `baselines/gaussian_regressor.py`, `baseline_models.py`, `simulation/linear_ah.py`, `gaussian_mlp_preflight.py` |
| 2 | #170 | `auto/contribution-gmlp-group-copula` | contribution | `simulation/linear_ah.py`, `contribution_gmlp_copula_rho.py`, two preflights, a diagnostic, a stamper |
| 3 | #172 | `auto/switch-joint-exodus-gmlp` | switch | the gnn `#171` files again (code-identical; docstrings differ) |
| 4 | #174 | `auto/switch-exodus-k-onehot` | switch | `generic/joint_exodus.py`, `generic/graph.py` |
| 5 | #177 | `auto/contribution-inflated-gmlp` | contribution | `baselines/gaussian_regressor.py`, `baseline_models.py`, `simulation/linear_ah.py`, `stamp_contribution_group_copula.py`, `gmlp_inflated_preflight.py` |

Out of scope: the 23 `[FAIL]` PRs (read their logs when they explain a
choice, never audit their code), anything merged into `main`
before #146, and the frozen surface (§8 of `notes/autoresearch.md`) — if the
review's finding is that a frozen file is wrong, that is an escalation,
not a finding to fix.

## 3. Depth: what the reviewer may and may not do

The review is **read-only and tooling-free**. Its value comes from
reading the diff against the architecture it lands in, not from
instrumenting it.

**May**: `git diff` / `git show` / `git log` across the branches, read
any file at any revision, read the PR bodies and
`notes/autoresearch_log/<slug>.md` (declaration, plan, notes), read the
committed `scores.csv` / results tables, read the tests a step added
(read them — they are evidence about intent), grep.

**May not**: train, simulate, evaluate, touch Raven, write a new script,
notebook, or helper of any kind, install anything, or modify an
experiment branch. Nothing this review produces is executable.

**The one exception**: when a **critical** finding turns on a number
that is already committed in the repo, one throwaway command line over
that file is allowed to settle it. It is quoted in the finding and never
saved as a file. If a question needs more than that, it is not chased —
it is written down as an open question with the one experiment that
would settle it, and the maintainer decides whether it is worth a run.

**Budget**: one pass per step, biggest-first within a step. A step's
review is done when every file it *created* has been read and every file
it *modified* has been read at the diff plus enough surrounding context
to judge fit. Helper scripts that only produced an artifact (fitters,
preflights, stampers) get read for what they *did to the numbers* — the
fit target, the data, the selection rule — not line by line.

## 4. Order: beginning to end

Each lineage is walked root first, in the order the steps were built.

- Every step was written against its parent and judged against its
  parent. Reading forward puts the reviewer where the agent that wrote
  it stood, with the same architecture in front of them and the same
  deficit profile to explain.
- **Fit is only judgeable against something already read.** `main` is
  the known-good architecture; every step after it is judged against the
  state its predecessors left behind. That state is available going
  forward and is guesswork going backward.
- Accretion is visible as it happens. `graph.py` is rewired by #165,
  #171, #174 and #179 and `linear_ah.py` by #160, #167, #170 and #177 —
  read forward, each edit is a layer going on, and the step where a file
  stops being coherent is obvious at the moment it stops.
- The tip is then read last, as what it actually is: the sum of
  everything before it.

The one thing forward order does not hand over for free is
**subsumption** — a step that a later step quietly made redundant. Two
places pay for it explicitly: §5.7, where every step names what it takes
over from its predecessors, and a **tip reconciliation** that closes
each lineage — with the whole chain read, list the mechanisms the tip
actually still needs, and anything added on the way that is not on that
list is a redundancy finding.

Both lineages are walked independently, gnn then gmlp, and §7's
cross-lineage pass comes last — twins are only visible once both sides
have been read.

## 5. The per-step procedure

For each step, in this order:

1. **Declaration vs diff.** Read the log file's declaration and the PR
   hypothesis, then the diff. Does the diff contain the declared change
   — and does it contain anything else? One experiment = one change
   (§4). Extra mechanism riding along is a finding whatever its merit.
2. **Read the new files.** Every file the step created, in full.
3. **Read the modified files at the diff, in context.** For
   `graph.py`, `linear_ah.py` and `train.py` this means reading enough
   of the surrounding function to judge whether the edit is a seam or a
   wedge.
4. **Trace one decision end to end.** Pick the step's central quantity —
   a sampled contribution, a switch decision, a punishment draw — and
   follow it from the training objective through the artifact to the
   simulation call. Most bugs and every train/sim parity break live on
   that path.
5. **Ask the four questions** (§1) explicitly, and write down the answer
   for each even when it is "clean".
6. **Architecture fit, in one paragraph.** Where does the mechanism
   live, where *should* it live, and what does the next experiment on
   this slot have to work around? This paragraph is required for every
   step; it is what makes the review useful beyond a defect list.
7. **Look back.** Name what this step takes over from its predecessors:
   a function it replaced, a code path it made unreachable, a config
   nobody runs any more, a fitter whose output is no longer consumed.
   Anything superseded and still present is a redundancy finding
   recorded here, at the step that superseded it.
8. **Record findings** per §6 and move to the child.

## 6. Findings: categories and severity

Every observation is one finding with a category, a severity, evidence,
and a consequence. No severity is assigned without stating what the
finding would change.

**Categories**

| tag | category | tells |
|---|---|---|
| `R` | redundancy | the same mechanism implemented twice; a later step subsuming an earlier one; parallel fitters / stampers / preflights; cloned configs; dead code a config no longer names |
| `B` | bug or cheat | train/sim divergence; seeding or masking errors; round-anchoring off-by-one (punishment conditions on t-1 only); information available at simulation time that the human did not have; silent fallbacks that turn a failure into a plausible number |
| `S` | spoon-feeding | a parameter fitted against evaluation output rather than human data; a constant hand-stamped onto an artifact; a feature keyed to a metric's definition (a bin edge, a stratum, round 1 being special); arm-shopping — several sim configs run, the friendliest reported; a control arm that quietly redefines the baseline |
| `X` | convolution / misfit | the mechanism patched into simulation instead of learned; scaffolding heavier than the change it delivers; a feature with exactly one consumer; accretion in a shared file that the next experiment will have to unpick |

`S` findings are graded against §5 of `notes/autoresearch.md`, which
permits selecting variants by evaluation score but forbids engineering
at a metric's definition and forbids stack-shopping. Say which side of
that line the finding falls on, and when it is genuinely on the line,
say that — the borderline cases are the ones the maintainer wants.

**Severity**

| severity | meaning |
|---|---|
| **critical** | if true, a reported result should be read differently: the verdict could be wrong, the gain does not come from the stated mechanism, or the code is wrong in a way that changes simulated behavior. Also: a §5 rule actually broken. |
| **mild** | real cost, no result in question — duplication, dead code, an unused feature, a convention violated, scaffolding that has to be maintained. |
| **cosmetic** | tidiness only — naming, file placement, docstrings, leftover debug paths, a config clone nobody reads. |

A finding that cannot be settled by reading is recorded as an **open
question** with the same tags plus the single check that would close it.
Open questions are never promoted to critical on suspicion.

**Output**: **one document per lineage**, named for the lineage it
reviews — `reports/autoresearch_review_gnn.md`,
`reports/autoresearch_review_gmlp.md` — plus
`reports/autoresearch_review_cross.md` for §7 once both exist. One
lineage is one review and can be spun off on its own; nothing else, no
per-step files. Structure (the cross file carries only §7 and the
combined judgement):

- a summary table, critical first: `id | step | tag | severity | finding
  | evidence | consequence`,
- one section per step in review order (root to tip, gnn then gmlp),
  each holding its findings in full plus the §5.6 architecture-fit
  paragraph,
- a tip reconciliation closing each lineage (§4): what the tip still
  needs, and what it therefore does not,
- the cross-lineage section (§7),
- a closing per-lineage judgement: adopt / adopt-after-fix / leave, with
  the findings that drive it.

Finding ids are `<tag>-<pr>-<n>`, e.g. `S-170-2`. Evidence is always
`path:line` at a named revision, or a quoted log line.

## 7. Cross-lineage pass

The lineages were developed in parallel and converge on the same ideas,
so the last pass compares them directly. Three twin pairs are known to
exist; the pass establishes for each whether the two sides are one
mechanism or two, and which implementation should survive:

- **joint exodus** — #171 (gnn) and #172 (gmlp) both add
  `generic/joint_exodus.py`, `generic/conditional_bernoulli.py` and the
  same `train.py` hook, differing by ~78 added / ~60 removed lines.
  Establish whether that diff is a real divergence before calling it
  one: compare the parsed syntax trees with docstrings stripped, not the
  line diff. (The 2026-09-17 pass found the code identical and the whole
  diff documentary; the divergence arrives at #174, which edits only the
  gmlp copy.)
- **contribution copula** — #165 (gnn, `generic/copula.py` plus a
  613-line fitter) and #170 (gmlp, a 905-line fitter plus two preflights
  and a diagnostic) implement the same group-latent idea at different
  layers and at very different scaffolding cost.
- **severity copula** — `scripts/baselines/punishment_copula_rho.py`
  (679 lines) is added by **both** roots, #160 and #167, byte-identical,
  and #167 is a contribution experiment. Its sim config
  (`23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch.yml`) names
  `punishment_multinomial_severity_copula.joblib` — the unmerged
  candidate from the other lineage — in a PR rooted at `main`. Establish
  from the log and the PR whether that stack was the maintainer's
  instruction (§3/§9 allow a parent-targeted stack) or a §5
  stack-shopping question. Report it as found; do not assume either way.

Also compare what each lineage did to the same shared file — `graph.py`
carries edits from #165, #171, #174 and #179, and `linear_ah.py`
from #160, #167, #170 and #177 — and say whether the two sets can coexist in
one `main` or whether adopting both means a merge that nobody has done.

## 8. Conventions

- Review work lives on its own branch, `autoresearch-review`, branched
  from `main`; one PR at the end. It adds this file and the per-lineage
  findings documents of §6, and nothing else. A lineage review spun off
  on its own still lands on that branch — separate documents, one PR.
- Experiment branches are never modified, rebased, or pushed to. A
  finding that implies a fix names the fix; it does not make it.
- A verdict recorded in a PR is never re-judged here. A finding may say
  "this result should be read differently" — that is the maintainer's
  call to act on, and the summary's numbers stay as they are.
- Findings are about the code and the process, never about the agent
  that wrote them. "The fitter selects rho on the evaluation output"
  is a finding; "the agent cheated" is not.
- Nothing in the review touches the frozen surface (§8 of
  `notes/autoresearch.md`) or reopens the summary's frozen corpus.
