# Autoresearch Review: the gnn Lineage

Code review of `main -> #160 -> #165 -> #171 -> #179` — the per-group
virtual-node stack, tip mean 1.099 — per the workflow in
`notes/autoresearch_review.md`. Read-only: no experiment branch was
modified, no experiment re-run, no verdict re-judged.

**Status:** complete, all four steps. Its sibling review of the
gaussian-MLP lineage (`main -> #167 -> #170 -> #172 -> #174 -> #177`)
is `reports/autoresearch_review_gmlp.md`; the cross-lineage pass (§7 of
the workflow) runs once both exist and lands in
`reports/autoresearch_review_cross.md`.

---

## Summary

**One critical finding**, in two instances: the copula's persistence
parameter φ was set to its maximum, 1.0, in both steps that use it —
once where the estimate was inadmissible and once where it was not —
and φ = 1 is the setting that most helps the target row. This is one
half of a campaign-wide pattern; the other half is #170's ρ_p/ρ_t
split on the gmlp lineage, and the two are read together in
`reports/autoresearch_review_cross.md` (C-X-1), which is where the
pattern is stated in full.

Nothing else here puts a reported verdict in question: the two largest
claimed moves (PD 2.93 → 1.53, CG 4.27 → 0.90) are multi-band and far
outside the sensitivity of anything below. The rest is maintenance
debt, one duplicated mechanism, and a defence that does not operate
where it was claimed to.

| id | step | tag | severity | finding |
|---|---|---|---|---|
| S-165-1 | #165 | spoon-feeding | **critical** | Pre-registered φ stop-gate fired and was resolved to the boundary φ = 1.0 |
| S-179-1 | #179 | spoon-feeding | **critical** | φ rule again resolves to 1.0 though φ̂ = 0.618 was admissible; φ = 1 is the dose that most helps the target |
| R-165-1 | #165 | redundancy | mild | Second Gaussian-copula sampler duplicates #160's; no shared code |
| B-171-1 | #171 | bug/cheat | mild | Conditional-Bernoulli defence is vacuous in exactly the cells the mechanism creates |
| X-179-1 | #179 | misfit | mild | Contribution module imports from the switch slot's module |
| R-171-1 | #171 | redundancy | mild | Two cluster-isolation mechanisms; one superseded by `main` |
| X-165-3 | #165 | misfit | mild | Cluster infrastructure shipped inside an experiment PR (§4) |
| X-171-1 | #171 | misfit | mild | Five cluster scripts rewired inside an experiment PR (§4) |
| R-179-1 | #179 | redundancy | mild | Five near-identical SLURM stamper wrappers |
| X-165-1 | #165 | misfit | mild | `n_groups = 2` hard-coded in the copula cell id |
| X-165-2 | #165 | misfit | mild | Copula path bypasses `y_encoder.decode`; safe only for onehot |
| B-165-1 | #165 | bug/cheat | cosmetic | `copula.py`'s stated RNG contract is false under full exodus into group 0 |
| B-160-1 | #160 | bug/cheat | cosmetic | `assert` used for bundle-field validation (stripped under `-O`) |
| X-165-4 | #165 | misfit | cosmetic | `copula.py` docstring still describes it as the switch sampler |
| X-179-2 | #179 | misfit | cosmetic | Tests land in `tests/vnode/`, siblings' in `src/aimanager/tests/` |

What the review cleared is as load-bearing as what it found, and is
recorded per step below — in particular the redundancy question this
lineage most invites (copula vs virtual node) is answered by the
campaign's own ablation, not by assertion.

---

## 1. #160 — severity copula, punisher slot

`auto/punisher-severity-copula-v2`, 3 files: a sim config, a 679-line
calibration script, `simulation/linear_ah.py` (+60/−7).

**Declaration vs diff.** Exact match. One pairing, one config, one
mechanism; the config differs from the reference stack only in the
punisher slot and the output dir.

**Traced decision.** `get_punishments` → `_pool_from_rounds` →
`_class_probs` → `_sample_levels_copula` → level. The sampler draws one
shared latent per group id, mixes `u = Φ(√ρ·z_g + √(1−ρ)·ε_i)`, and
inverts through each agent's own CDF. Marginals are preserved by
construction, and the group latent is taken from the same round dict the
features come from (`linear_ah.py:341`). That dict carries the *current*
round's membership (`simulate.py:264`), which is the partition the
evaluation attributes punishments to — the right cell. Feature legality
holds: the punishment target admits only the `prev_` family
(`notes/baseline_feature_defs.md:12`), enforced at config validation.

**Cleared — the spoon-feeding question.** ρ is a pairwise-likelihood MLE
of an exchangeable Gaussian copula on the bundle's own locked train
split. The optimisation is over the *likelihood*, not over any score
(`punishment_copula_rho.py:220`). Split diagnostics are printed under a
header that reads "diagnostic splits only (never a selection criterion)"
(`:624`). The pre-flight does compute the PD statistic itself (`:426`,
commented as mirroring `evaluation_suite/metrics._spread_ratio`) — the
single most metric-adjacent thing in the lineage — but it selects
nothing; it only forecasts whether a simulation is worth spending. The
reason this stays on the legal side of §5 is structural, not
incidental: ρ is fixed by MLE before the pre-flight runs.

**Cleared — bundle discipline.** `save_bundle` asserts no pre-existing
key was modified by identity (`:479`) and re-checks `predict_proba`
bit-identity after a reload (`:485`). `bvn_cdf` is validated against
scipy's mvn CDF at five ρ values before use (`:183`).

**B-160-1 (cosmetic).** `copula_rho` range and applicability are
validated with bare `assert` in `LinearAHAdapter.__init__`
(`linear_ah.py:91-104`). Under `python -O` a malformed bundle would
sample silently instead of failing.

**X-160-1 (mild).** The mechanism is a special-cased branch inside
`LinearAHAdapter`, gated to `is_punishment and model_type ==
"multinomial"`. It is correct and well-guarded, but it is not an object
another slot can use — which is the direct cause of R-165-1.

**Architecture fit.** Good. The change is confined to the sampler, adds
one optional bundle field, and leaves the independent path — including
its exact RNG consumption — untouched when the field is absent. A
reader of `linear_ah.py` can see the whole mechanism in one screen. The
cost is that the copula is expressed as *punisher plumbing* rather than
as a sampling primitive, so the second user had to start over.

**Protocol note (cleared for this lineage).** #160 is the only step
whose baseline comes from the 4-pairing reference run while the
candidate runs a single pairing. Pairings execute sequentially inside
one process against one global seed (`simulate.py:736-743`, then the
`for name, run in runs.items()` loop at `:186`), so a candidate's stream
matches the baseline's only if the baseline pairing is *first* in the
reference config. Here it is (`lin_multinomial_self` leads). Model
loading does not disturb this — `GraphNetwork.load` passes the pickled
submodules back into `__init__`, which rebuilds nothing, and the
encoders are parameter-free. The invariant holds but is asserted
nowhere, and it is the same invariant whose violation produced the
known config-dependence on PR #162. #165, #171 and #179 all compare
single-pairing run against single-pairing run and are unaffected.

## 2. #165 — herding copula, contribution slot

`auto/contribution-herding-copula-v2`, parented on #160. New
`generic/copula.py` (106), `generic/graph.py` (+84/−1), three
calibration/stamping scripts, two test suites, one sim config, and one
SLURM launcher.

**Declaration vs diff.** Match, with one rider: `scripts/simulate_iso.slurm`
(93 lines of cluster plumbing) is not part of the declared change.

**R-165-1 (mild) — the same mechanism, implemented twice.**
`copula.sample_correlated_levels` is mathematically the same sampler
as #160's `LinearAHAdapter._sample_levels_copula`: shared latent per cell,
`√ρ`/`√(1−ρ)` mix, inverse-CDF through each row's own cumsum, identical
`searchsorted` convention and clamp. #160's version is a strict special
case (`z_prev=None`). Nothing is shared between them; `linear_ah.py` was
not refactored onto the new module. There is a real reason unification
is not free — the two consume the RNG differently (#160 draws
`randn(n)` twice and picks the group latent by first-member index for
composition stability; #165 draws `randn(n_cells)` then `randn(N)`), so
merging them would change the punisher's stream and break its
bit-identical reproduction of PR #146. That reason is nowhere written
down, and the next experiment on this axis wrote a *third* copy on the
gmlp side. **Consequence:** at the tip, a change to copula sampling has
two independent implementations to make and keep consistent.

**S-165-1 (critical) — the φ boundary ruling.** The plan pre-registered a
stop-gate: `phi_hat >= 1 → no artifact, no simulation, escalate` (log
`:162`). It fired — φ̂ = 1.1588, CI [0.826, …]. The escalation resolved
by adopting φ = 1.0, the unit-root boundary, and by relaxing the two
`phi < 1` asserts to admit it (log `:219`, `:249`). This is defensible:
the estimate left the admissible space, projecting a constrained MLE
onto its boundary is standard, the CI's lower end is far from 0, and the
ruling was recorded before the simulation ran. It is worth recording
anyway, because φ = 1.0 is both the maximum admissible persistence and —
as #179's later sweep proves — the dose that most helps the target row.
No interior-φ control arm was run here. **Consequence:** the reported
CG 9.81 → 4.16 is a two-band move that no plausible dose choice
explains away, so the verdict stands; but the effect *size* is reported
at the most favourable admissible dose. See S-179-1, which is the same
rule applied to a non-saturating estimate.

**X-165-1 (mild).** `cells = batch_index * 2 + agent_group`
(`graph.py:379`) hard-codes two groups. Self-flagged in the log's note
2. Silently wrong, not loud, if a config ever changes group count.

**X-165-2 (mild).** The copula path returns raw level indices and never
calls `y_encoder.decode`, which the legacy path uses
(`graph.py:367`). This is correct *only* because the head's encoder is
`IntEncoder(encoding="onehot")`, where decode-with-sampling returns the
column index itself. The guard that exists checks `y_name`, not the
encoding, so a future head with `encoding="numeric"` would silently
return indices where the legacy path returns scaled values. The log's
own step-4 check (b) verifies the equivalence for today's head; nothing
enforces it.

**X-165-3 (mild).** `scripts/simulate_iso.slurm` is shared cluster
infrastructure. §4: a shared-code change is legal but is *its own
experiment*. See R-171-1 for what it collides with.

**B-165-1 (cosmetic).** `copula.py`'s RNG contract states that "neither
rho, phi, z_prev nor the cell composition changes what is consumed"
(`copula.py:68-76`). It does: consumption is `randn(n_cells)` with
`n_cells = cell_id.max() + 1`, so a round in which every agent sits in
group 0 draws one latent instead of two. The `graph.py` caller knows
this and handles it correctly (`:398`, and the comment at `:404-407`
about a group emptying out); only the module's own contract is wrong.

**X-165-4 (cosmetic).** `copula.py`'s header still reads "the herding
sampler for the GNN switch predictor" and points at the switch log,
though the module is now the contribution sampler too.

**Cleared — leakage and split discipline.** The calibration asserts no
train/test episode overlap *and* that no flip-copy of a train game sits
in the holdout (`contribution_copula_rho.py:330-333`) — a direct guard
on the §5 illegality about flipped duplicates. The stamper verifies the
source model's sha256, refuses a `phi_final` without a
`phi_final_reason`, and asserts only the three new fields were added.
One sim config, one arm.

**Architecture fit.** The dispatch is clean — one `if sample and
copula_rho > 0` in `predict_independent`, three optional constructor
kwargs, three new keys in `save`. An artifact without the fields loads
and behaves exactly as before. The misfit is not in the wiring but in
the layering: a sampling primitive that both the linear and the graph
model need now exists twice, once as a method and once as a module, and
the module still carries the identity of the slot that first wrote it.

## 3. #171 — joint exodus head, switch slot

`auto/switch-joint-exodus`, parented on #165. New `generic/joint_exodus.py`
(283) and `generic/conditional_bernoulli.py` (185),
`artificial_humans/train.py` (+197/−11), `graph.py` (+217/−2), a
498-line train/sim parity suite, two configs — and seven cluster
scripts.

**Declaration vs diff.** The modelling change matches the declaration
precisely and adds no input feature. The seven cluster scripts do not
belong to it.

**Traced decision.** Pooled post-RNN embeddings per (batch, round, group
label) → MLP → logits on a 9×9 count grid → `-inf` on `m_g > k_g` before
a single `log_softmax` → joint draw of `(m_0, m_1)` → conditional
Bernoulli picks *which* members leave. The mask leaves `(0,0)` valid for
every non-negative `k`, so the grid is never entirely masked. Subset
weights accumulate as sums of log-odds and normalise through one
`log_softmax` over all 2⁸ subset codes — exact conditional Bernoulli by
enumeration, not a sequential approximation.

**B-171-1 (mild) — the defence does not cover the cells the mechanism
creates.** Conditional-Bernoulli selection is introduced explicitly as
"a defence of the individual-level rows that PR #168's shared latent
broke … never as a lever on SC" (log, *What is inherited*). But when
`m == k` there is exactly one subset of size `m`, so the per-agent
probabilities have no influence on who leaves — the defence is vacuous
in full-exodus rounds. The joint head's entire purpose is to make those
rounds more frequent (the 5 → 8 transition is the named target). So the
protection is weakest precisely where the mechanism is strongest.
**Consequence:** RCD and assortativity are defended in the rounds where
nothing was at risk and undefended in the rounds the head adds. This is
a property of the design, provable by construction, not a defect in the
code — but it qualifies the claim made for the conditional-Bernoulli
half.

**X-171-1 (mild).** Five cluster scripts (`train_cluster.sh`,
`simulate_cluster.sh`, `fetch_cluster.sh`, `remote_test.sh`,
`run_simulation.sh`) were rewired for `AI_REMOTE_DIR` isolation inside
this experiment PR. §4 requires a shared-code fix to be its own
experiment with its own before/after.

**R-171-1 (mild).** `main` has since landed an equivalent isolation
mechanism independently (`SBATCH_EXPORT=ALL` plus a command-line
`PYTHONPATH`). Four of the five scripts are now byte-identical between
this branch and `main`, so most of X-171-1 has neutralised itself. What
remains at the tip is two orphans: `run_simulation.sh` carries an
in-template `export PYTHONPATH="$PWD/src…"` (+8 lines) that duplicates
what `main` already injects, and #165's `simulate_iso.slurm` (93 lines)
does not exist on `main` at all. **Consequence:** adopting this lineage
means deciding between two isolation designs, not merging one.

**Cleared — the detach, and why it is the right call.** The joint head
reads a *detached* pooled embedding (`joint_exodus.py:249`), documented
with the measurement that drove it: attached, the joint term ran at ~2–3
nats against the per-agent term's ~0.5 and cost the per-agent model
held-out log-loss 0.5158 → 0.5200. Detaching makes the candidate "the
base model's trunk plus a head", so score movement is attributable to
the mechanism rather than to a re-fitted representation. A loss weight
was considered and rejected. This is exactly the discipline that makes
a result readable.

**Cleared — no new observable.** The head consumes pooled embeddings,
both valid-decider counts, and the round, all on the encoder's own
normalisation conventions. The refactorisation is of the *label*
distribution.

**Recorded, not a finding.** Gate 2 passed by margin rather than by
improvement: the mean *rose* 0.0147 (+1.14%) against the amended 10%
allowance. That is within the rule as amended and is disclosed; it is
the only step in this lineage that does not also improve the mean, and
it is why the tip's mean is 1.099 rather than lower.

**Architecture fit.** The two new modules are the best-fitting code in
the lineage: torch-only, unit-testable on macOS, no `torch_geometric`
import, and each carries a docstring that states its load-bearing design
points rather than its API. `pool_by_group` in particular was written
general enough that #179 reused it unchanged — the one place where this
lineage did *not* duplicate itself. The misfit is entirely outside the
model: an experiment PR that also re-plumbs the cluster launchers.

## 4. #179 — per-group virtual node, contribution slot (the tip)

`auto/contribution-group-vnode`, parented on #171. New
`generic/group_vnode.py` (208), `graph.py` (+101/−6), a stamper fix,
three SLURM wrappers, three sim configs.

**Declaration vs diff.** Match. The three configs resolve cleanly: the
candidate, plus a `nocopula` ablation and a `phihat` arm that the log
attributes to a **maintainer-requested sweep dated the day after the
verdict** (note 26). The results table carries exactly two rows,
baseline control and candidate. This is not arm-shopping; it is an
ablation published with its negative parts intact.

**S-179-1 (critical) — the φ rule, applied to a non-saturating
estimate.**
The pre-declared rule (carried from #165) resolves "CI includes 1" to
φ_final = 1.0. Here φ̂ = 0.6182 with CI [0.2116, 1.2637] — *admissible*,
mean-reverting, and nowhere near the boundary. The rule still gave 1.0
and the implementer applied it mechanically. The log's note 16 is
candid about it: "a rule written for a saturating estimate is now being
applied to one that is not", kept because "pre-declaration binds" and
because the CI is too wide to distinguish persistence levels, with "a
successor should test φ̂ directly on this trunk … it is not claimable
here." What the review adds is the measured consequence, from the
campaign's own sweep (note 26):

| | φ = 1 (adopted) | φ = 0.618 (estimated) | no latent |
|---|---|---|---|
| CG | **0.8990** (≤ 1) | 1.2217 (1-2) | 2.4007 (2-5) |
| RCD | **1.3405** | 1.8511 | 2.3248 |
| mean | **1.0988** | 1.1156 | 1.2429 |

The adopted dose is the one that best serves the target row, and the
estimated dose would have cost CG a band. **Consequence:** the headline
"CG at the noise ceiling" is a property of *the virtual node at maximum
admissible persistence*, not of the virtual node. The pre-registration
is genuine and the disclosure is complete; the finding is that the rule
itself resolves uncertainty in the direction that helps, and it has now
done so twice.

**X-179-1 (mild) — cross-slot import.** `group_vnode.py` imports
`N_GROUPS`, `SIZE_NORM` and `pool_by_group` from `joint_exodus.py`. The
contribution slot's module therefore cannot be adopted without the
switch slot's experiment file. Reusing the pooling is right; the
location is wrong. **Fix:** lift `pool_by_group` and the two constants
into a neutral module both slots import.

**R-179-1 (mild).** Five SLURM wrappers now exist across #165 and #179
(`calibrate_copula`, `stamp_copula`, `calibrate_copula_group_vnode`,
`stamp_copula_group_vnode`, `stamp_copula_group_vnode_phihat`). The
phihat wrapper differs from its sibling in six lines — three job-name /
log-path lines and two artifact paths. One parameterised script taking
`BASE`/`PARAMS`/`OUT` would replace all five.

**X-179-2 (cosmetic).** Tests land in `tests/vnode/` while the GNN tests
of #165 and #171 live in `src/aimanager/tests/`. `tests/baselines/`
predates the campaign, so a top-level `tests/` is not itself new — but
the lineage now splits its test tree by step rather than by kind.

**Cleared — the redundancy question this lineage most invites.** Do the
copula and the virtual node do the same thing twice? Both inject
group-level shared structure into contributions. The campaign measured
the overlap rather than assuming it away: recalibrating on the retrained
trunk drops ρ from #165's 0.0696 to 0.0436 — the trunk absorbed part of
the shared component, exactly as the plan predicted it would — and the
`nocopula` ablation puts CG at 2.4007 without the copula against 0.8990
with it. They overlap and are not redundant, and the size of the overlap
is on the record. This is the standard the other redundancy findings
above should be read against.

**Cleared — RNG and stamping discipline.** The virtual node is
constructed *last* of all submodules so that with `group_vnode=False`
no RNG is drawn and the model stays bit-identical to the parent's
(`graph.py`, build-order comment). Stamping was verified to leave 14
tensors bit-identical including the GRU's own weights, which is what
licenses attributing the result to the node rather than to the stamp.

**Architecture fit.** The node is a trunk change and is honest about
being one: trained attached, with the log explaining why the
`JointExodusHead` detach does not apply (there is no competing loss).
Membership is read at the round the state is broadcast, so a switcher
picks up the arrival group's state — the mechanism and the RCD claim are
the same object. Emptied groups are handled as a real state rather than
an error. The one structural cost is the widening of `op2` by the group
state, which is confined and documented. The fit problem is X-179-1: the
module reaches across a slot boundary for a utility.

---

## Tip reconciliation

With the whole chain read, what the #179 tip still needs:

| from | still needed at the tip |
|---|---|
| #160 | `_sample_levels_copula` + the ρ fitter — yes, the punisher slot runs on it |
| #165 | `copula.py`, the `graph.py` dispatch, the ρ/φ stamping recipe — yes (the ablation shows the copula still carries CG) |
| #165 | the `..._herding_copula_v2` **artifact** — no: #179 recalibrated ρ on its own retrained trunk and ships its own |
| #165 | `simulate_iso.slurm` — no: superseded by `main`'s mechanism |
| #171 | `joint_exodus.py`, `conditional_bernoulli.py`, the `train.py` hook — yes, and doubly so since #179 imports `pool_by_group` from it |
| #171 | four of the five cluster-script edits — no: byte-identical to `main` already |
| #171 | `run_simulation.sh`'s `PYTHONPATH` export — no: duplicates `main`'s `SBATCH_EXPORT=ALL` |
| #179 | `group_vnode.py`, the `graph.py` wiring, one stamper — yes |
| #179 | the `phihat` / `nocopula` configs and the third stamper — not for the tip; keep as the published ablation |

Everything on the "no" rows is already covered by a finding
(R-171-1, R-179-1). Nothing else added along the way is unreachable at
the tip — which, for a four-step stack built by four separate
experiments, is a good result.

## Judgement: gnn lineage

**Adopt after fixes**, and the fixes are small.

The modelling code is the strongest part of this lineage. Three of the
four steps ship modules that are torch-only, unit-testable off-cluster,
guarded by asserts at their real preconditions, and documented with
their load-bearing design decisions rather than their signatures. Two
decisions in particular — #171's detach, backed by a measured log-loss
cost, and #179's build-last RNG ordering — are the kind of discipline
that makes a result attributable. The empty-group state, the flip-copy
leakage guard, and the bit-identity checks on stamping are all handled
where a less careful lineage would have produced a plausible number.

Before adoption:

1. **Unify the copula samplers** (R-165-1), or, if the RNG divergence
   makes that too costly, write the reason down at both sites.
2. **Move `pool_by_group`** out of `joint_exodus.py` into a neutral
   module (X-179-1) so the contribution change stands alone.
3. **Decide the isolation mechanism** (R-171-1): keep `main`'s and drop
   `simulate_iso.slurm` plus the `run_simulation.sh` export, or state
   why both survive.
4. **Collapse the five SLURM wrappers** into one parameterised script
   (R-179-1).

Two things to carry forward rather than fix:

- The **φ rule** (S-165-1, S-179-1) is critical, and is half of the
  campaign-wide pattern set out in the cross-lineage review's C-X-1: a
  persistence parameter the data does not pin down has been resolved to
  its maximum every time it has come up, across both lineages, and the
  maximum is every time the setting that most helps the target row. The
  justification on offer — PR #150's arm comparison — says the
  persistent latent *scored* better, not that human group culture is
  undecaying; #179's own estimate says the opposite (φ̂ = 0.618,
  substantial mean reversion) and the model was given none. Rewrite the
  rule before it is used a fourth time, and run the successor #179's
  note 16 already names.
- The **conditional-Bernoulli defence** (B-171-1) does not operate in
  full-exodus rounds. Any future claim that rests on it should say so.

And one invariant worth asserting rather than relying on: the §3
baseline comparison is stream-aligned only because the baseline pairing
leads the reference config. It holds for #160 and does not arise for the
rest of this lineage, but it is the same invariant that produced the
config-dependent draw on PR #162.
