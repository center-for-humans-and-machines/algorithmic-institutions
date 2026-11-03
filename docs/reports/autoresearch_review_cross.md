# Autoresearch Review: Cross-Lineage Pass

§7 of `notes/autoresearch_review.md`, plus the combined judgement. Reads
the two completed lineage reviews against each other:
`reports/autoresearch_review_gnn.md` (the chain `main -> #160 -> #165
-> #171 -> #179`, tip mean 1.099) and
`reports/autoresearch_review_gmlp.md` (`main -> #167 -> #170 -> #172
-> #174 -> #177`, tip mean 1.1249).

The question this pass exists to answer: **the two lineages converge on
the same ideas and never merge — are they one mechanism or two, and can
both tips land in one `main`?**

---

## Summary

| id | tag | severity | finding |
|---|---|---|---|
| C-X-1 | spoon-feeding | **critical** | Whenever the data could not pin down how persistent the shared latent is, it was set to the maximum — three times, both lineages, always the setting that most helps the target |
| R-X-1 | redundancy | mild | One genuine merge (`graph.py`) and one duplicate file to delete; everything else is additive or docstring-only |
| R-X-2 | redundancy | mild | Two training configs share one path and differ only in `output_dir` |
| R-X-3 | redundancy | cosmetic | Band upgrades counted per PR are a gross figure; on both spines the gross count is roughly double the net |

The headline result of this pass is **negative in the useful sense**:
the twin pairs the workflow flagged are far less entangled than they
looked, and adopting both tips is a much smaller job than the raw
diffstat suggests.

---

## 1. The three twin pairs

### Joint exodus (#171 gnn / #172 gmlp) — one mechanism, one merge

The workflow seeded this as "a diverged copy, not a shared module",
inferred from a ~78/−60 line diff. **That inference was wrong.** Parsing
both revisions and comparing the syntax trees with docstrings stripped:

| file | #171 vs #172 | gnn tip vs gmlp tip |
|---|---|---|
| `generic/joint_exodus.py` | code identical | `__setstate__` + `size_encoding` (gmlp only) |
| `generic/conditional_bernoulli.py` | code identical | code identical |
| `artificial_humans/train.py` | code identical | code identical |

At the twin step the two stacks ran **the same code**; every changed
line was documentation (rewordings, added clarifications, a correlation
re-measured for the gmlp stack at −0.1987 against the gnn stack's
−0.4676, and retargeted log references). The divergence arrives one step
later, at #174, and is a deliberate additive option — `size_encoding`
defaults to `"numeric"` and `__setstate__` supplies that default when
unpickling a head saved before the option existed.

**Verdict: one mechanism, and the merge is to take the gmlp version.**
Its default reproduces the gnn tip's behaviour exactly, so nothing on
the gnn side has to change. Two of the three files need no merge at all.

### Contribution copula (#165 gnn / #170 gmlp) — not twins

These are not two implementations of one idea. They are two different
objects at two different layers:

| | #165 | #170 |
|---|---|---|
| emission | categorical (21-way softmax) | Gaussian `N(mu(x), sigma(x))` |
| latent → level | CDF inversion | correlated normal taken directly |
| persistence | AR(1), `phi` | persistent + transient split, `rho_p`/`rho_t` |
| host | `GraphNetwork` | `LinearAHAdapter` |

They share only the idea of mixing a shared normal with idiosyncratic
noise at `sqrt(rho)` weights. The genuine duplicate is elsewhere and
internal to the gnn lineage: #165's `copula.sample_correlated_levels`
against #160's `_sample_levels_copula`, both categorical with CDF
inversion (R-165-1 in the gnn review). #177 later *consolidates* on the
gmlp side rather than adding a fourth, generalising #170's sampler so
only the final normal → level step branches on the emission family.

**Verdict: two mechanisms, correctly separate.** What differs
indefensibly is the scaffolding around them — a 613-line fitter on one
side against 905 + 534 + 466 + 302 lines on the other (X-170-1).

### Severity copula (#160 gnn / #167 gmlp) — one file, arriving twice

`scripts/baselines/punishment_copula_rho.py` is **byte-identical at both
tips** (confirmed: the file does not differ between the two
branches). #167 obtained it by an explicit, credited `git checkout` from #160's
branch, under a maintainer directive to root at `main` but use #160's
punisher. The sampler half rides inside `linear_ah.py`, where the gmlp
tip is a strict superset (below).

**Verdict: one file, delete one copy at adoption.** No merge.

## 2. Can both tips land in one `main`?

Ten files are touched by both lineages. Nine differ between the tips.
Classified by what a merge actually costs:

| class | files | cost |
|---|---|---|
| code identical (docstrings only) | `conditional_bernoulli.py`, `train.py` | pick either |
| byte identical | `punishment_copula_rho.py` | delete one copy |
| strict superset (gmlp) | `simulation/linear_ah.py` | take gmlp's |
| additive option (gmlp) | `generic/joint_exodus.py` | take gmlp's |
| genuine union merge | `generic/graph.py` | **the one real merge** |
| infrastructure, superseded by `main` | `run_simulation.sh`, `run_training.sh` | drop (gnn review R-171-1) |
| same path, different content | `switch_predictor/joint_exodus.yml` | rename one (R-X-2) |
| test suite | `test_joint_exodus_train_sim_parity.py` | follows its module |

**`linear_ah.py` is a strict superset on the gmlp side.** Five shared
methods differ and no method exists only on the gnn side; every
difference is a generalisation that preserves the gnn tip's behaviour —
`_sample_levels`'s multinomial test widens from `== "multinomial"` to
`in _CATEGORICAL`, which contains it; `__init__` gains `rho_p`/`rho_t`
fields gated to the Gaussian contribution sampler; `_reset_history`
gains a latent store. Any bundle the gnn tip can load behaves
identically under the gmlp file.

**`graph.py` is the one real merge — and the conflicts are textual, not
semantic.** Six shared definitions differ; `_predict_encoded_copula`
exists only on the gnn side. The clearest illustration is what each tip
persists in `GraphNetwork.save`:

| | persisted fields beyond the 13 shared |
|---|---|
| gnn tip | `copula_rho`, `copula_phi`, `copula_switch_every` (#165); `joint_exodus`, `joint_exodus_head`, `joint_exodus_switch_every` (#171); `group_vnode`, `group_vnode_module`, `group_vnode_hidden` (#179) |
| gmlp tip | `joint_exodus`, `joint_exodus_head`, `joint_exodus_switch_every` (#172); `joint_exodus_size_encoding` (#174) |

The union is well defined: **no key appears on both sides with a
different meaning**, and the joint-exodus triple is identical. The
merge conflicts land in `__init__` (both add constructor kwargs), `save`
(both extend one list), and `forward`/`encode`/`predict_independent`
(each adds a disjoint path). A careful union, not a reconciliation of
competing designs.

**R-X-1 (mild).** So the adoption cost is: one union merge in
`graph.py`, one file deleted, one config renamed, two infrastructure
orphans dropped. That is materially smaller than the ~3,000-line raw
diff between the tips implies, and it is worth recording because the
opposite impression — two entangled forks — is what the diffstat gives.

**R-X-2 (mild).** `configs/training/artificial_humans/switch_predictor/joint_exodus.yml`
exists on both lineages with the same path and different content. The
functional difference is one line — `output_dir:
artifacts/artificial_humans/switch_joint_exodus` against
`..._switch_joint_exodus_gmlp` — everything else is comment prose. Two
training configs that produce two different artifacts cannot share a
path; one needs the lineage suffix its artifact already carries.

## 3. C-X-1 (critical) — persistence is always set to the maximum, and
the maximum is always what wins

Both lineages model group culture as a shared latent whose *persistence*
is a free parameter. Both had to choose that parameter three times
between them. The data never pinned it down, and every time it was
resolved to the most persistent value available:

| step | parameter | what the data said | adopted | the alternative |
|---|---|---|---|---|
| #165 | φ (AR(1) on the latent) | φ̂ = 1.1588 — outside the admissible range | **1.0** | boundary projection; forced, and defensible on its own |
| #179 | φ, re-estimated on the new trunk | φ̂ = 0.6182, CI [0.212, 1.264] — admissible, mean-reverting | **1.0** | 0.618, the point estimate, ignored because the CI touches 1 |
| #170 | ρ_p / ρ_t split | unidentified; lag-1 share ~0.006 with a CI spanning zero | **fully persistent** (ρ_t = 0) | the round-by-round reading, which its own log calls "the honest two-component reading" |

**The adopted value is, every time, the one that most helps the target
row.** #179's own sweep measures it: CG 0.8990 at φ = 1 against 1.2217
at the estimated φ̂ — a full band, on the target, bought by the setting
the data did not support. #170's preflight predicted CG ~2.8 for the
persistent reading against ~5.3 for the alternative, i.e. pass against
fail.

**And the justification bottoms out in "it won."** The reason #179 gives
for keeping 1.0 over an admissible 0.618 is that "PR #150's arm
comparison found the persistent latent wins and the fast-reverting one
regresses". That is a statement about *scores in a previous experiment*,
not about human behaviour. Nobody measured persistence in the human data
and found it saturating; #179's own estimator says the opposite, and the
model was given no mean reversion at all. §5 permits selecting variants
by evaluation score — but what is being selected here is a structural
assumption about the data-generating process, chosen the way a
hyperparameter is chosen.

**What defends it, stated fully.** Each rule was pre-registered before
the estimate was visible, so no individual step exercised post-hoc
discretion; #165's estimate genuinely was inadmissible, where projecting
onto the boundary is standard; #170's falsifier was written down in
advance and the run came back on the predicted number; and #179's note
16 flags the tension itself — "a rule written for a saturating estimate
is now being applied to one that is not" — and says a successor should
test φ̂ directly, "it is not claimable here". The authors saw this. It
is disclosed in every log.

**Consequence.** It is nonetheless the case that the campaign's frontier
stack runs on a persistence setting that won rather than one that was
measured, and that the single largest target-row gain in the whole
campaign (CG to the noise ceiling at #179) is a band better *because of*
that setting. The pattern is what makes it critical: one boundary
projection is a judgement call, three in the same direction across two
independent lineages is a rule that is biased toward the target.

**What would settle it:** two simulations, roughly 2.5 min each — #179's
trunk at φ̂ = 0.618 (already stamped and run once as an ablation, so only
the reporting changes) and #170's two-component arm at
(ρ_p ≈ 0.006, ρ_t ≈ 0.021), which has never been simulated. If the
measured values hold their bands, the pattern is harmless and the rule
can be rewritten to adopt the point estimate. If they do not, the
frontier's headline numbers are dose artefacts and should be restated.

## 4. R-X-3 (cosmetic) — band upgrades are counted gross, not net

*Maintainer ruling, 2026-09-17: recovering ground a previous step lost
is not a problem worth flagging. Recorded here as a reporting note
only.* Both lineage reviews found steps that band-downgrade rows earlier
steps had upgraded. Measured across both spines:

| spine | step-by-step upgrades | step-by-step downgrades | net root → tip | rows ≤ 1 |
|---|---|---|---|---|
| gnn (`main` → #179) | 12 | 7 | **6 up, 4 down** | 11 → 12 |
| gmlp (#167 → #177) | 16 | 7 | **10 up, 2 down** | 8 → 11 |

(The gmlp row starts at #167's candidate; #167's own step adds one more
downgrade against its pre-lineage baseline — S-167-1.)

**On both spines the gross upgrade count is roughly twice the net.** §2
is what produces this: a step is judged only on a band upgrade of a
*declared target* plus the 21-row mean, and there is no term for a band
downgrade on any other row. So a row can be upgraded, lost, and upgraded
again, and the record shows two successes.

The sharpest single instance is C-170-2 in the gmlp review: #167 moved
CG from 3.9781 to 5.9106 (downgrade, on the row it had itself named as
the guard), and #170 then claimed a `> 5 -> 2-5` band upgrade back to
2.8292. Net across the pair, CG is `2-5` at both ends. The gnn lineage
has a quieter version: #171 downgraded RCD 1.965 → 2.765, which #179
then recovered to 1.340, and #171's SB left `<= 1` and never returned.

**Consequence.** Per-PR band upgrades — the leaderboard's ranking
criterion and the campaign summary's narrative unit — are a gross
figure. They are correct as a record of what each experiment achieved
against its own parent, which is what §2 asks. They are not a measure of
how far the stack moved, and the two differ by about a factor of two.
Nothing here says a verdict is wrong; the finding is that the aggregate
should be reported net, or reported as gross with the net alongside.

**What would settle the presentation:** the net columns above are
computable from the committed `spine_scores.json` with no new runs, and
the campaign summary already renders per-row trajectories. Adding a net
band-change count beside the gross one is a reporting change, not a
research one.

## 5. Combined judgement

**Adopt both, gmlp-first, after the four gnn fixes and the four gmlp
fixes already listed in the two lineage reviews.**

The two lineages are not competing forks of one design; they are a
graph model and a tabular model that borrowed from each other in one
direction (gmlp took #160's punisher and #171's switch head) and never
collided on purpose. Taking the gmlp tip's `linear_ah.py` and
`joint_exodus.py` wholesale, then merging the gnn tip's `graph.py`
additions in as a union, reproduces both tips' behaviour — the defaults
and the `save` key sets are disjoint by construction, which is a sign
both lineages were written with an eventual merge in mind even though
neither was asked to be.

On code quality the two are close and their strengths differ. The gnn
lineage writes the better modules — torch-only, off-cluster testable,
each carrying its load-bearing design decisions in prose, with #171's
measured detach and #179's build-last RNG ordering the two best
attribution decisions in the campaign. The gmlp lineage runs the better
protocol — #167 re-established its baseline in the candidate's own
single-pairing shape instead of reading it out of a multi-pairing run
(the invariant the gnn review found unasserted), and #177 both hardened
a silent wrong-model fallthrough into a raise and declined to bank a
band upgrade whose mechanistic test had failed.

Three things to carry past adoption, none of which is a defect in any
PR:

1. **Settle the persistence parameter** (C-X-1, and its instances
   S-165-1, S-179-1, C-170-1). This is the one critical finding of the
   review, it spans both lineages, and it is closed by two ~2.5 min
   simulations. Until then the frontier runs on a setting that won
   rather than one that was measured, and the campaign's single largest
   target-row gain depends on it. **Rewrite the rule** so an admissible
   point estimate is adopted rather than overridden by a wide interval
   that happens to touch the boundary.
2. **One claimed defence does not cover what it is claimed to cover.**
   Conditional-Bernoulli selection is vacuous when `m == k`, which is
   exactly the cell the joint head exists to produce (B-171-1).
3. **Report band movement net as well as gross** if the campaign
   summary is ever read as a measure of stack progress (R-X-3) —
   cosmetic under the maintainer's ruling, and a reporting change
   rather than a research one.

The critical finding is not a code defect and does not put a recorded
verdict in question — it says the frontier's headline dose was chosen
for winning rather than for being measured, and names the two cheap runs
that would resolve it. Across nine experiment PRs, roughly 6,000 lines
of new modelling and analysis code, and two independently developed
stacks, that is the substantive result of this review.
