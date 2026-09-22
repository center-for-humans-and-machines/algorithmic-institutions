# Autoresearch Review: the gaussian-MLP Lineage

Code review of `main -> #167 -> #170 -> #172 -> #174 -> #177` — the
inflated-emission stack, tip mean 1.1249 — per the workflow in
`notes/autoresearch_review.md`. Read-only: no experiment branch was
modified, no experiment re-run, no verdict re-judged.

**Status:** complete, all five steps. Its sibling review of the gnn
lineage is `reports/autoresearch_review_gnn.md`; the cross-lineage pass
(§7 of the workflow) is still to come in
`reports/autoresearch_review_cross.md`.

---

## Summary

**One critical finding**, C-170-1: the persistent/transient split
of #170's shared latent was not identified by the data and was resolved to
*fully persistent* — the reading its own preflight predicted would
pass, against an alternative it predicted would fail. It is the third
instance of a campaign-wide pattern (the gnn lineage's φ = 1.0 is the
other two), stated in full as C-X-1 in
`reports/autoresearch_review_cross.md`.

No finding here is a code defect, and none puts a recorded verdict in
question.

| id | step | tag | severity | finding |
|---|---|---|---|---|
| C-170-1 | #170 | spoon-feeding | **critical** | The ρ_p/ρ_t split was unidentified and resolved toward the arm predicted to pass; the alternative reading was pre-predicted to fail |
| R-174-1 | #174 | redundancy | mild | Edits only the gmlp copy of `joint_exodus.py`, diverging the two tips' copies of one module |
| X-170-1 | #170 | misfit | mild | 1,781 lines across four standalone preflight/diagnostic scripts that nothing imports |
| R-167-1 | #167 | redundancy | mild | #160's fitter and sampler ported verbatim onto a main-rooted branch |
| S-174-1 | #174 | spoon-feeding | mild | One-hot over the variable SC measures — cleared, with a residual small-sample concern |
| S-177-1 | #177 | spoon-feeding | mild | Atoms at 0 / 20 / prev sit exactly where CF and RCA measure — cleared |
| X-172-1 | #172 | misfit | cosmetic | A 138-line diff against its twin that is entirely docstrings |
| C-170-2 | #170 | redundancy | cosmetic | The credited CG upgrade recovers a band its own parent #167 lost one step earlier |
| S-167-1 | #167 | spoon-feeding | cosmetic | CG was the pre-declared guard, was breached, and cost nothing because §2 gates only the target and the mean |

**A correction to my own workflow doc.** §7 of `notes/autoresearch_review.md`
seeded #171/#172 as "a diverged copy, not a shared module", inferred
from a ~78/−60 line diff. That inference was wrong. Comparing the
parsed syntax trees with docstrings stripped, `joint_exodus.py`,
`conditional_bernoulli.py` and `train.py` are **code-identical**
between #171 and #172. The divergence is real but arrives later, at #174, and is
a deliberate feature rather than drift.

---

## 1. #167 — gaussian MLP v2, contribution slot

`auto/contribution-gaussian-mlp-v2`, rooted at `main`. Three sim/training
configs, `baselines/gaussian_regressor.py` (+69/−3),
`baseline_models.py` (+40/−11), `simulation/linear_ah.py` (+64/−11), a
430-line preflight, and a verbatim copy of #160's 679-line fitter.

**Declaration vs diff.** Match, and the two things that look irregular
in the diff are both directed and documented. The branch is rooted at
`main` yet its sim config names
`punishment_multinomial_severity_copula.joblib` — #160's unmerged
candidate from the *other* lineage. This is the §7 question the workflow
seeded, and it resolves cleanly: the declaration records a maintainer
directive of 2026-09-01 ("branch from `main`; integrate the PR #160
copula punisher into the stack; update the baseline metrics first"), and
plan step 2 is an explicit `git checkout origin/auto/punisher-severity-copula-v2 --`
of four named paths with credit to #160. **Not stack-shopping.** The
baseline was re-established by a fresh simulation of that exact stack
before the candidate ran, so both gates measure only the
contribution-slot delta.

**Cleared, and better than its sibling lineage.** That fresh baseline
run is built from a purpose-made config with the pairings reduced to
one. This is precisely the stream-alignment invariant I flagged as
unasserted on the gnn side (`autoresearch_review_gnn.md`, #160's
protocol note): rather than read a baseline out of a multi-pairing
reference run, #167 re-ran the baseline in the same single-pairing shape
as the candidate. On this axis the gmlp lineage is the more careful of
the two.

**S-167-1 (cosmetic, same ruling) — the guard was breached and it cost
nothing.** The
declaration names CG explicitly as "the named guard: it is what killed
PR #151 through gate 2", excluded as a target precisely because it was
the known risk. CG then went **3.9781 → 5.9106**, a band downgrade
(`2-5` → `> 5`). The experiment passed anyway: gate 1 asks only for a
band upgrade on a *declared target* (RCA 5.208 → 4.000, genuine), and
gate 2 asks only that the mean not rise (1.6308 → 1.6145, it fell). §2
contains no term for a band downgrade on a non-target row. **Consequence:**
the step is a legitimate `[SUCCESS]` that nonetheless left the stack
worse on the row its own author had identified as the thing to protect,
and the framework had no way to register that. Fully disclosed in the
results table. See C-170-2 for what happens next.

**R-167-1 (mild).** `scripts/baselines/punishment_copula_rho.py` (679
lines) and the `copula_rho` block in `linear_ah.py` now exist
byte-identically on two independent branches. This is the documented
consequence of "root at `main` but use the other lineage's artifact" —
the branch needed the fitter to regenerate the bundle. Deliberate, not
careless; but at adoption it is one file arriving from two directions.

**Architecture fit.** The `gaussian_mlp` support is three string
changes to a dispatch that already existed — the smallest possible
footprint for a new emission. The feature core is *forced in* by
declaration rather than selected by CV, which is the right way round: the
PR #151 post-mortem blamed a "CV-shopped 5-feature set", and this step
pre-registers the behavioural core and searches only around it. All
features are prev-family and `validate_feature_legality` hard-errors
otherwise.

**Look back.** Nothing to supersede — first step of the lineage.

## 2. #170 — group copula on the gaussian sampler, contribution slot

`auto/contribution-gmlp-group-copula`, parented on #167.
`simulation/linear_ah.py` (+108/−1), a 905-line fitter, a 534-line
preflight, a 466-line diagnostic, a 302-line stamper, one config.

**Declaration vs diff.** Match. One arm, one config.

**C-170-1 (critical) — an unidentified split, resolved toward the
passing arm.** The estimator identifies the *total* residual
within-cell dependence well (moment share 0.0263, CI [0.0157, 0.0376]).
It does not identify how that total divides between a **persistent**
per-(episode, group) latent and a **transient** per-(round, group) one:
the lag-1 cross-member share is ~0.006 with a **CI spanning zero**. The
log states both readings, and its own pre-run predictions for them:

| reading | ρ_p | ρ_t | predicted CG |
|---|---|---|---|
| "the honest two-component reading" (the log's phrase) | ~0.006 | ~0.021 | **~5.3 — a FAIL** |
| "the persistent reading" | ~0.028 | 0 | **~2.8 — a band upgrade** |

The shipped bundle carries ρ_p = 0.04378520865574197, **ρ_t = 0.0** —
the persistent reading. So a structural choice the data could not make
was made in the direction that the campaign's own pre-flight predicted
would pass the gate, while the arm the log itself calls "honest" was
predicted to fail.

*What defends it, stated fully.* The choice and the falsifier were
recorded **before** the simulation ran, explicitly "so the verdict
cannot be re-read after the fact", with the discriminating prediction
quantified: near 6.6 means the transient reading was right and the
structural choice was wrong; inside `2-5` means the persistent reading
was right. CG landed at **2.8292** — close to the persistent reading's
~2.8 and nowhere near ~5.3 or ~6.6. That is a sharp, quantitative,
pre-registered prediction that came true, which is real evidence for the
choice. The gnn lineage corroborates it independently (PR #150's arm
comparison and #179's φ sweep both find persistent beats fast-reverting).

*What remains.* The transient arm was never simulated, so the
discrimination rests on a preflight proxy rather than on two runs of the
real stack. **Consequence:** #170's CG result should be read as "CG
upgrades *given* that the shared component is persistent" — a conditional
the leaderboard entry does not carry. **The one check that would close
it:** simulate the two-component arm (ρ_p ≈ 0.006, ρ_t ≈ 0.021) on this
stack. One sim, ~2.5 min on Raven by the log's own budget note.

**C-170-2 (cosmetic, maintainer ruling 2026-09-17) — the credited
upgrade is a return leg.** Recorded because it is the clearest instance
of how §2's per-step accounting reads in aggregate; the maintainer's
ruling is that recovering points a previous step lost is not a problem
worth flagging, and that governs. #170's
primary declared target is CG at **5.9106**, band `> 5`. That is exactly
the value #167 left behind after moving CG from 3.9781 (S-167-1). The
lineage's CG trajectory:

| | pre-lineage baseline | #167 | #170 | net |
|---|---|---|---|---|
| CG | 3.9781 (`2-5`) | 5.9106 (`> 5`) | 2.8292 (`2-5`) | **`2-5` → `2-5`** |

Every step is correct under §3, which mandates judging against the
parent's confirmed scores. But read end-to-end, the pair records one
band downgrade and one band upgrade on the same row in the same slot in
consecutive steps, and the campaign's aggregate — the leaderboard's
band-upgrade counts, the summary's narrative — counts only the upgrade.
**Consequence:** the lineage's genuine CG progress is 3.9781 → 2.8292, a
real 1.15 improvement *within* a band, not the band upgrade the record
shows. No verdict is wrong and no rule was broken; the accounting is
what produces the impression.

**X-170-1 (mild) — the preflight estate.** This step adds a 534-line
preflight and a 466-line diagnostic, on top of #167's 430-line preflight
and before #177's 351-line one: **1,781 lines across four standalone
scripts** at the tip, none imported by anything in `src/`, each
re-implementing the same scaffolding (load bundle, rebuild features,
replay sampling, compute a target statistic). The gnn lineage does the
same job with a `--preflight` flag on the calibration script it already
had. Preflighting itself is right — §5's iteration budget makes a wasted
simulation expensive — but the form is four one-shot programs rather
than a flag.

**Cleared — the third sampler is not a third copy.**
`_sample_levels_gaussian_copula` shares only the latent-mixing idea
with the categorical samplers of #160 and #165; a Gaussian emission takes the
correlated normal directly, with no CDF inversion, which is a genuinely
different object. It also preserves #160's RNG invariant explicitly —
"exactly 3n float64 draws per call, in the fixed order zu, zv, eps,
taken unconditionally … so a switch cannot shift the stream and arms
stay comparable" — and handles the re-forming-group case deliberately (a
group that empties and returns resumes its own latent).

**Architecture fit.** Good. The marginal-preservation argument is stated
in the docstring as an algebraic identity rather than asserted, and it
names its own bug-detector: "a large [RCA] move means the marginals are
not preserved". That is the right way to make a sampler falsifiable from
its output. The misfit is entirely in the surrounding analysis estate.

**Look back.** Supersedes nothing of #167's; the trunk is unchanged and
the copula is stamped on top.

## 3. #172 — joint exodus head on the gmlp stack, switch slot

`auto/switch-joint-exodus-gmlp`, parented on #170. The #171 files again:
`joint_exodus.py` (288), `conditional_bernoulli.py` (195), `train.py`
(+200/−11), `graph.py` (+213/−2), a 544-line parity suite, two configs.

**X-172-1 (cosmetic), and the workflow's §7 seed corrected.** Measured
against #171, this step's three shared modules show a 138-line diff. Parsing both
revisions and comparing the syntax trees with docstrings stripped, the
**executable code is identical** in all three files. Every changed line
is documentation: rewordings, added clarifications (that `agent_group`
is the sole membership signal, that `k` counts deciders not members), a
correlation re-measured for *this* stack (−0.1987 pooled over decision
rounds, against the gnn stack's −0.4676), and log references retargeted
to this branch. So the cross-lineage duplication is a genuine copy of a
file, but it carried **no divergence risk** at this point — the two
stacks ran the same code.

Whether the port should have been a shared module rather than a copy is
a real question, but it is a question about branch topology (two
lineages that never merge) more than about this step, and it belongs in
the cross-lineage pass.

**Cleared.** The measured between-group correlation being weaker on this
stack (−0.1987 vs −0.4676) is reported rather than elided, and the
mechanism is claimed on that stack's own number.

**Architecture fit.** Same as #171's: the two modules are torch-only,
unit-testable off-cluster, and carry their load-bearing design points in
prose. The detach ("THE CUT") survives the port intact.

**Look back.** Takes over nothing; the contribution slot is untouched.

## 4. #174 — one-hot group sizes in the joint head, switch slot

`auto/switch-exodus-k-onehot`, parented on #172. `joint_exodus.py`
(+57/−8), `graph.py` (+43/−2), one training config, two sim configs.

**S-174-1 (mild) — borderline, and cleared.** The change one-hot-encodes
**group size**, which is the variable SC — the declared target — is
defined on ("size of the larger group"). That is close to §5's
prohibition on "features engineered at a metric's definition". It
clears, for three reasons worth recording:

- the encoding spans `k ∈ {0..8}`, the action space's natural range, not
  SC's strata — SC is an EMD over a distribution and has no bin edges to
  key to;
- group size was *already* an input as the scalar `k / 8`; this changes
  its encoding, not its presence, and no new observable enters;
- the motivation is a measured human regularity, not a metric: `P(full
  exodus | k)` is 0.161 / 0.147 / 0.200 / 0.177 for `k = 1..4` and **0 of
  119 cells** for `k >= 5`, a hump and then a hard floor that one smooth
  scalar cannot bend through, against a stack emptying singletons in
  46.5% of cells versus the human 16.1%.

*The residual concern.* Nine free intercepts per group label let the head
encode "no group of five or more ever empties" as near-deterministic
structure, and that fact comes from 119 cells with zero positive
examples. SC is exactly the row that rewards getting that tail right. The
gain is real but rests on a small-sample zero being promoted to a
constraint.

**R-174-1 (mild).** This step edits **only the gmlp copy** of
`joint_exodus.py`. Verified: at #171/#172 the two copies were
code-identical; at the two tips they are not. **Consequence:** adopting
both lineages now requires merging a real code change into one file,
where before #174 it was a delete-one-copy operation.

**Cleared — backward compatibility.** `size_encoding` defaults to
`"numeric"`, and `__setstate__` supplies the default when unpickling a
head saved before the option existed, so previously trained heads keep
their 23-wide MLP and sample bit-identically. That is the right
discipline for a change to a pickled module's shape.

**Cleared — the cost was pre-declared.** The declaration states CG's
return to `2-5` as the price of the SC move. CG did move 1.968 → 2.079
(a boundary graze across 2.0, unlike #167's 3.98 → 5.91). Declaring the
price in advance is what S-167-1 did not do.

**Architecture fit.** Small, focused, and confined to the head it
changes. It is the cleanest single step in either lineage.

**Look back.** Supersedes the numeric size scalars of #172's head — but
by option rather than by replacement, so the earlier path stays reachable
and tested.

## 5. #177 — inflated contribution emission, contribution slot (the tip)

`auto/contribution-inflated-gmlp`, parented on #174.
`gaussian_regressor.py` (+279/−5), `baseline_models.py` (+113/−11),
`linear_ah.py` (+83/−17), the fitter (+151/−43) and stamper (+132/−42),
a 351-line preflight, three configs.

**Declaration vs diff.** Match, and the "is the copula re-estimation a
second change?" question is confronted rather than dodged: ρ is
model-conditional (fitted against the bundle's own teacher-forced
marginal), so re-estimating it on a new trunk is the recipe's derived
step, "as PRs #173 and #175 ruled". Reasonable, and consistent with the
way #179 handled the same question on the gnn side.

**S-177-1 (mild) — borderline, and cleared.** The emission gains
explicit probability atoms at `c = 0`, `c = 20` and `c = prev_contribution`.
Those are, respectively, the two values **CF** measures ("the share of
contributions equal to 0 and the share equal to 20") and the zero-change
point that **RCA**'s strata are built around. Three atoms placed exactly
where three rows look is the most metric-adjacent modelling choice in
either lineage. It clears:

- 0 and 20 are the action space's boundaries; a mass point at the
  boundary of a censored variable is standard, and CF exists *because*
  the boundaries are behaviourally special, not the reverse;
- the atom masses are **learned per observation** — the MLP head emits
  atom logits alongside `mu` and `log_sigma`, softmaxed against a body
  logit pinned at 0 — not hand-set to match a target share;
- the atom *subset* is selected by **CV cross-entropy** on held-out data,
  from a fixed allowlist, not by evaluation score.

The selection criterion is the decisive part: §5 permits choosing
variants by evaluation score, and this step did not even need that
permission.

**Cleared — and the best single piece of discipline in the campaign.**
CG band-upgraded here too (2.079 → 1.842), and the log **declines to
claim it**: "claimed on RCA; CG's crossing is real but not the declared
corner-locking cause, U3(a) having failed at 0.0443 against its
pre-registered 0.05". The mechanistic test for *why* CG should move
failed, so the row's movement is logged as collateral rather than
banked as a second success. That is pre-declaration binding in the
direction that costs the author something.

**Cleared — the sampler consolidates rather than proliferates.** #177
does not add a fourth copula. It generalises #170's
`_sample_levels_gaussian_copula` so the shared latent is constructed
once and only the final normal → level step branches on the emission
family: `_GAUSSIAN` rounds the normal, `_INFLATED` pushes it through the
discrete CDF. The dispatch is made exhaustive and raises on anything
unknown, with the comment naming the exact silent failure it prevents —
an inflated bundle inherits `predict`/`predict_std`, which return its
mixture's *body* parameters, so falling through to the Gaussian
arithmetic "would quietly run the incumbent emission and raise nothing".
Converting a silent wrong-model bug into a loud error is precisely the
kind of thing this review exists to look for, and here it was done
unprompted.

**Architecture fit.** The emission's surface changes from Gaussian
(`predict`, `predict_std`) to categorical (`predict_proba`, `classes_`),
which is a real interface shift; the dispatch above is what makes it
safe. The one structural cost is that `gaussian_mlp_inflated` still
*inherits* the Gaussian accessors it no longer means, which is why the
guard was needed — a cleaner hierarchy would not expose them at all.

**Look back.** Supersedes #170's Gaussian sampling path for this slot
(the tip's contributor is inflated, so `_GAUSSIAN` is now reachable only
by the older bundles), and supersedes #170's atom-free emission entirely.
Both remain reachable and tested, by option rather than by deletion.

---

## Tip reconciliation

With the whole chain read, what the #177 tip still needs:

| from | still needed at the tip |
|---|---|
| #160 (via #167) | the severity-copula punisher bundle and its fitter — yes, the punisher slot runs on it |
| #167 | the `gaussian_mlp` trunk, features and dispatch — yes, #177's emission is built on it |
| #167 | `gaussian_mlp_preflight.py` — no: one-shot, imported by nothing |
| #170 | the shared-latent construction in `linear_ah.py` — yes, #177 generalises it rather than replacing it |
| #170 | the `_GAUSSIAN` sampling branch — not for the tip's own contributor, which is `_INFLATED`; kept as the path older bundles take |
| #170 | the two 1,000-line preflight/diagnostic scripts — no: one-shot, imported by nothing |
| #172 | `joint_exodus.py`, `conditional_bernoulli.py`, the `train.py` hook — yes, the switch slot runs on them |
| #174 | `size_encoding="onehot"` — yes, it is the shipped switch head's setting |
| #177 | the inflated emission, the extended sampler, the re-fitted copula — yes |

Everything on a "no" row is covered by X-170-1. Nothing else is
stranded: the superseded paths (`_GAUSSIAN`, `size_encoding="numeric"`,
atom-free emission) all survive as tested options behind defaults, which
is a deliberate pattern in this lineage and a good one.

## Judgement: gaussian-MLP lineage

**Adopt after fixes, with the CG claim re-stated.**

The code is strong and in two places better than its sibling lineage:
the #167 baseline was re-run in the candidate's own single-pairing
shape rather than read out of a multi-pairing run, and #177 turned a
possible silent wrong-model fallthrough into a hard error unprompted.
The pre-registration discipline is real and occasionally costly to its
own authors — #177 declining to bank a CG band upgrade because its
mechanistic test failed is the single best moment in either lineage's
record.

Before adoption:

1. **Simulate the two-component arm** (ρ_p ≈ 0.006, ρ_t ≈ 0.021).
   This is the critical item (C-170-1) and the cheapest one: a single
   ~2.5 min simulation either confirms that the split the data could
   not make is the one the mechanism needs, or shows the choice never
   mattered. Until it runs, #170's CG result is conditional on an
   assumption rather than on a measurement.
2. **Re-state #170's CG claim** with that conditional attached. It is
   not hidden — the log states both readings and their predictions —
   but it does not reach the leaderboard.
3. **Reconcile `joint_exodus.py`** across the two tips (R-174-1) — a
   real merge now, not a delete-one-copy.
4. **Prune or fold the preflight estate** (X-170-1): 1,781 lines that
   nothing imports, doing a job the gnn lineage does with a flag.

One thing to note rather than fix (cosmetic per the 2026-09-17
ruling): **§2 has no term for a band
downgrade on a non-target row** (S-167-1). #167 breached its own
declared guard by a full band and passed on the mean; #174 crossed CG
back over 2.0 but pre-declared the price; #177 crossed SC and RCD back
out of `<= 1` while claiming RCA. On both spines this is common — the gnn
spine carries 7 band downgrades across 4 steps, this one 8 across 5, so
neither lineage is the outlier — but it means "band upgrades" counted
along a spine are a gross figure, not a net one. That belongs in the
cross-lineage pass and, ultimately, in how the campaign summary reports
progress.
