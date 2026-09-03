# Joint exodus: a round-level head over both groups' leaver counts

## Declaration

**Slot:** switch.

**Parent:** PR #165 (`auto/contribution-herding-copula-v2`), the maintainer-designated
frontier, at `4be127a`. PR opens with `--base auto/contribution-herding-copula-v2`.

**Predecessor:** PR #169 (`auto/switch-exodus-count`), a `[FAIL]` on this same
branch point. Its preflight is this experiment's foundation: read its log
(`notes/autoresearch_log/switch-exodus-count.md`) before touching anything here.
Its launcher fix is cherry-picked as this branch's first commit.

**Base model:** the GNN switch predictor
`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
(`x_encoding = common_good, punishment, agent_group, round_number`;
`edge_encoding = []`; `y_levels = 2`; no copula fields, so the legacy independent
`th.multinomial` draw).

**Evaluation stack (§3 under the parent rule of §9):** the parent's own stack and
config, `23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch`, single
pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
`save_per_round: true`.

**Baseline (the parent's confirmed scores; both §2 gates judged against these):**

| quantity | value |
|---|---|
| SC | 2.1867905905576634 (band 2-5) |
| mean over 21 rows | 1.2893632310269196 |
| rows <= 1 | 11/21 |

**Target row:** SC, 2.1867905905576634, band 2-5 -> requires 1-2 or better.
**Guard (gate 2):** the 21-row mean must fall below 1.2893632310269196.
**Declared watch item:** SB. PR #169 note 11 established that any mechanism which
pushes the simulation into more unbalanced states raises the realised switch rate
in the late decision rounds; SB is the row that pays for it.

### Hypothesis

A migration wave has a direction. When the minority group empties, the majority
simultaneously holds — the two groups' decisions in a round are opposed, not
independent. This is the same collective-exodus behaviour PR #169 hypothesised,
but at the boundary the data actually puts it: **between** the two groups rather
than **within** one.

PR #169 measured it. In the human data the two groups' leaver counts in the same
(episode, round) carry a residual correlation of **-0.4676** after controlling for
both group sizes and the round (game-clustered 95% CI [-0.5759, -0.3602],
permutation p < 0.0001, 148 complete pairs across 45 games), and still **-0.3038**
after conditioning on *both* groups' full round-level observable state. It is not
game-level heterogeneity: conditioning on each game's own switch rate makes the
correlation more negative, where heterogeneity would bias it positive. Raw, the
smaller group empties while the larger holds entirely in 14/137 = 0.1022 of pairs
against 0.0596 under independence, and P(larger holds | smaller empties) = 0.538
against 0.261 otherwise.

**The preflight for this experiment has already been run, and it passed as an
oracle.** Resampling the pair (m_smaller, m_larger) jointly from the human cells —
adding between-group dependence and nothing else, on the same data on which the
per-group arm reached only 5.7273 — gives mean larger-group size **6.1078** against
the human 6.0880, with EMD against the human size distribution collapsing from
0.361 to **0.052** and the defining 5 -> 8 transition landing at 0.172 against the
human 0.157. A round-specific variant independently lands at 6.0663. Against the
independent-null-to-human gap of +0.7203, per-group count arms carry 44-55%, the
method-of-moments correlation arm 57%, and the joint arm **103%**.

**Planned change.** Replace the per-group factorisation that PR #169 foreclosed
with a **round-level joint head over both groups' leaver counts**. At each decision
round the switch GNN pools its node embeddings separately for each group (masked by
`agent_group`), and a head reads the two pooled vectors together with both group
sizes and the round out into a joint distribution over the pair (m_0, m_1),
masked to m_g <= k_g. At simulation time each round draws the pair from that head,
then draws *which* members leave within each group from the **conditional
Bernoulli** induced by the existing per-agent probabilities. The per-agent head is
untouched and still decides who.

**Why a joint head is the right object, and not more conditioning.** The residual
dependence survives conditioning on every observable a pooled-embedding head could
see (-0.3038). That is not a failure of feature engineering to be fixed with better
features — it is irreducible residual dependence, and a joint distribution over the
pair is precisely the object that represents it. A head emitting two conditionals
p(m_0 | state) and p(m_1 | state) and drawing them independently can only induce
the correlation of the conditional means, which the regressions above already
absorb. This is why the per-group design failed at an oracle bound and why this one
is not the same experiment with a bigger network.

**What is inherited from PR #169 rather than rebuilt.** Two results transfer and
must not be re-litigated. A count head is provably moment-consistent with the
per-agent marginals — the maximum discrepancy between E[m | k, round] under the
empirical counts and k * p(switch | k, round) is 2.220e-16 across every cell — so
fitting counts by likelihood cannot by itself move SA or SB. And conditional-
Bernoulli selection of which members leave is exactly neutral for SC, confirmed
analytically (the state updates as a <- a - m_0 + m_1, in which mover identity
appears nowhere) and numerically across a 2x range of propensity heterogeneity. It
is kept as a defence of the individual-level rows that PR #168's shared latent
broke (assortativity 0.278 human vs 0.110 under the copula; RCD band-degraded in
all four arms), never as a lever on SC.

**Not a copula, and not by relabelling.** The five prior switch experiments all
dosed a Gaussian latent with a hand-calibrated correlation parameter, uniformly
across cells and rounds. This head is fitted by maximum likelihood on observed
labels, conditions on state so that size and round dependence is learned rather
than imposed, and being categorical over the pair it can represent the unanimity
structure and the opposition between groups that a single scalar cannot. The
mechanism PR #168 named as its successor — an anti-correlated between-label
latent — is here replaced by a learned joint whose sign and magnitude are measured
(-0.4676) rather than assumed.

**Rows this should move.** SC, by restoring the opposed-movement transitions that
build the missing right tail. SA and SB are the watch items per the moment-
consistency result and note 11's warning. RCD is the row the conditional-Bernoulli
half exists to protect.

## Plan

Validated by the orchestrator against §2, §5 and §8 before any step ran. No step
adds an input feature: the model consumes exactly what it consumes today, and the
only new learned object is a factorisation of the *label* distribution, which is a
likelihood choice. Nothing on the frozen surface is touched; the simulation
protocol, seeds and episode count are the parent's.

1. **The joint exodus head** — `src/aimanager/generic/graph.py`, `GraphNetwork`.
   Add an optional round-level head: pool the post-RNN node embeddings separately
   per group (mask by the `agent_group` column of the state, not by the edge index,
   which is group-agnostic), concatenate the two pooled vectors with both group
   sizes and the round, and read out logits over the pair (m_0, m_1) on a padded
   9 x 9 grid, masked to m_g <= k_g before a single softmax over the flattened
   grid. Gate the head behind a constructor argument defaulting to off and persist
   it through `save()`/`load()` alongside the existing `copula_*` fields, so an
   artifact without it loads exactly as today. The pooling must be order-canonical
   in the group index, not in size, so that the flip-doubled training data
   symmetrises the head rather than fighting it. *[Opus]*

2. **The joint training objective** — `src/aimanager/artificial_humans/train.py`
   plus a new config
   `configs/training/artificial_humans/switch_predictor/joint_exodus.yml`.
   Derive the per-(episode, decision round) leaver-count *pair* from the existing
   `does_switch` labels under the `switch_valid` mask, and add its cross-entropy to
   the existing per-agent loss. Handle the validity wrinkle explicitly: 109 of
   2,000 human decision rows fail `switch_valid`, so in 84 of 465 group cells the
   number of valid deciders is below the true group size and the two sizes do not
   sum to 8 — decide and document whether incomplete pairs are dropped or carried,
   and state the resulting n. The config copies `opt_50ep_doubled_reanchored.yml`
   unchanged (375 epochs, batch 10, lr 5e-4, hidden 10, 5-fold, seed 38381,
   flip-doubled data per the GNN convention) and only enables the head. Report both
   loss components per fold. *[Opus]*

2b. **Detach the head from the trunk** (plan revision, §9 step 4; ruled after
   step 2 reported, before step 3 was dispatched) —
   `src/aimanager/artificial_humans/train.py`. Stop the joint loss's gradient at
   the pooled embeddings, so the shared trunk is optimised by the per-agent loss
   alone and the head is fitted as a readout on top of it. Verify that the
   resulting per-agent held-out log-loss matches a head-off run of the same
   config and seed, and report both. *[Opus]*

   **Why the plan changed.** Step 2 measured what the plan had assumed away: the
   joint term is ~2-3 nats against ~0.5 for the per-agent term, so it dominates
   the trunk's gradient by sheer magnitude, and the local single-fold run showed
   the cost landing exactly where this experiment cannot afford it — held-out
   per-agent log-loss 0.5200 with the head against 0.5158 without. SB is the
   declared watch item and gate 2 is what killed PR #168; degrading the per-agent
   switch model to buy a better joint fit is the same trade in a new costume.
   Detaching removes it by construction: the trunk is trained by exactly the
   objective the base artifact was trained by, so the candidate is the parent's
   switch model plus a joint head, and any score movement is attributable to the
   mechanism rather than to a re-fitted representation. This also matches the
   declared hypothesis, which is about the sampling structure between groups, not
   about better representations. The cost is that the head can no longer shape the
   embeddings it reads — acceptable, because the head receives both group sizes
   and the round directly, and the dependence it exists to capture is by
   definition the part that observables do not explain, so it lives in the shape
   of the joint rather than in the features. Adding a loss *weight* instead was
   rejected: a weight is a tuning knob that would have to be searched, and
   searching it before any evaluation score exists is how one change per
   experiment becomes several.

3. **The conditional-Bernoulli sampler** — new
   `src/aimanager/generic/conditional_bernoulli.py`, with unit tests that run
   locally (no PyG import). Given a group's per-agent probabilities and a count m,
   draw the leaving subset exactly by enumerating the C(k,m) <= 70 subsets with
   weight proportional to the product of odds. Tests assert: marginals recovered by
   averaging over m drawn from a binomial match the input probabilities; propensity
   ordering is preserved; m = 0 and m = k are handled. *[Sonnet]*

4. **Wire the joint draw into the simulation** — `graph.py`, alongside the existing
   copula dispatch in `predict_independent`. On decision rounds with the head
   present and `sample=True`, replace the independent `th.multinomial` draw with:
   per-agent probabilities from the existing head, a pair (m_0, m_1) drawn per
   (batch element, round) from the joint head, then the step-3 subset draw within
   each group. Non-decision rounds keep running the forward pass to hold the GRU
   warm and must consume no extra RNG, matching current semantics. *[Opus]*

5. **Train/sim parity test** — `src/aimanager/tests/`. Assert that the group
   pooling and the count pair derived at training time from `data.py` match what the
   simulation builds from `environment.py` on the same synthetic membership,
   closing the untested pandas-vs-torch invariant for this mechanism. *[Sonnet]*

6. **Train on Raven** — `scripts/train_cluster.sh ah <config>` with
   `AI_REMOTE_DIR='~/autoresearch/switch-joint-exodus'`. The base run is 3m32s and
   the §5 budget ceiling is ~10.5 min; record actual elapsed time and the in-job
   artifact sha256. *[Sonnet]*

7. **Baseline control, then the candidate** — two simulations from the isolated
   dir. First re-run the parent's own config unchanged and require a bit-identical
   `per_round.parquet` (sha256 `4f64fc42...` per PR #168's control), which is the
   licence to compare anything; then the candidate, a three-edit copy of the
   parent's config differing only in `switch_model`, `output_dir` and
   `figure_name`. *[Sonnet]*

8. **Evaluate and rule** — `python -m aimanager evaluate` on the candidate,
   locally. One simulation, one evaluation, no second stage (§3). Record the results
   row, then open the PR titled `[SUCCESS]` only if SC leaves band 2-5 *and* the
   mean falls below 1.2893632310269196; otherwise `[FAIL]`. Report SA and SB
   explicitly whatever the verdict. *[Opus]*

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-09-02 | round-level joint head over the leaver-count pair (m_0, m_1), detached from the trunk, with conditional-Bernoulli selection of who leaves | single | **SC 1.147392663266986** (baseline 2.1867905905576634) | 11/21 (baseline 11/21) | 1.3040409569053069 (baseline 1.2893632310269196) | **SUCCESS** — gate 1 passes (SC 2-5 -> 1-2); gate 2 passes under the amended §2 (mean 1.3040409569053069 against a 10% ceiling of 1.4182995541296117, a rise of 1.1384%) |

## Notes

1. This experiment exists because PR #169's stop condition fired. The per-group
   count factorisation was foreclosed at an *oracle* bound — 5.7622 against a
   declared 5.99, with the human count distribution handed over for free — so no
   amount of training could have rescued it. The diagnostic that killed it measured
   the between-group dependence and resampled it, which is why this successor opens
   with its preflight already passed rather than pending.
2. The joint head's ceiling is the joint resample's 6.1078, and that number is an
   in-sample bootstrap of 167 pairs with several strata under 35 observations. It
   partly memorises the human data and overshoots reach-8 (0.511 against 0.420), so
   it is an optimistic ceiling, not a forecast. The honest expectation is that a
   learned head lands below it; the gate is a band upgrade on SC, not a match of
   the oracle.
3. Step 2's ruling on the timeout artefact is worth carrying beyond this
   experiment: on a group whose members did not all submit a decision,
   `m_g == k_g` does not mean the group emptied. 18 of 112 apparent full-exodus
   cells (16%) are timeouts, and carrying incomplete pairs inflates
   P(apparent full exodus | k_g > 0) from 0.1079 to 0.1204. Any future work on
   the exodus tail — the 5 -> 8 transition, the unanimity spike — must drop
   incomplete pairs or it will fit an artefact. Note that the between-group
   correlation itself is indifferent to the choice (-0.3660 on complete pairs
   against -0.3658 on all), so this is a bias in the tail statistics only, not
   in the dependence the head is fitted to.
4. The joint loss is unweighted, and step 2b's detach is what makes that safe.
   Before the detach, the joint term's ~2-3 nats against the per-agent term's
   ~0.5 dominated the trunk gradient and cost 0.0042 nats of held-out per-agent
   log-loss. If a successor ever re-attaches the head, a loss weight stops being
   optional.
5. Local end-to-end training under PyG stand-ins (CPU, single fold, not the
   experiment) showed the head learning under the shipped hyperparameters:
   joint loss 3.006 -> 1.887 nats over 375 epochs and still descending at the
   end. The head is not starved by the 66.8% of pairs that survive the drop
   ruling, but the fact that it has not converged is a note for a successor,
   not a licence to change the epoch count here — the config is deliberately
   byte-identical to the base run's.
6. **The isolation tooling was missing entirely and had to be ported.** PR #165
   forked from `main` at `4b2f3f4`, before `8680db9` and `849e3ca`, so
   `AI_REMOTE_DIR` was not implemented on this branch at all — the launchers
   hardcoded the shared checkout, and setting the variable would have silently
   fired `rsync --delete` into `~/algorithmic-institutions`. Neither tree alone
   carried a correct SLURM template: `main` rewrote the activation line to honour
   `AIMANAGER_VENV` but has no in-job `PYTHONPATH` export, while this branch's
   step 1 added the `PYTHONPATH` block but not the venv indirection. The union of
   the two is what actually works, and is what commit `fbec309` lands.
7. **Why the isolation design is sound at all, measured rather than argued.** The
   shared venv installs the package through a plain path `.pth`
   (`_editable_impl_algorithmic_institutions.pth`, one line:
   `/raven/u/certuer/algorithmic-institutions/src`), not an
   `__editable___*_finder` meta-path hook. A `sys.path` dump under
   `PYTHONPATH=<iso>/src` puts the isolated `src/` at index 0 and the shared
   checkout's editable path at index 5, appended by `site.addsitedir` after
   `PYTHONPATH` is already in place. Had setuptools chosen the import-hook form,
   it would sit on `sys.meta_path`, which is consulted *before* `sys.path`, and
   every "isolated" run in this project's history would have been importing
   shared code with no symptom — including the runs re-executed specifically to
   repair that bug.
8. **`main` carries the training-side twin of the PR #166 hazard, in latent
   form.** `scripts/artificial_humans/run_training.sh` on `main` has no in-job
   `PYTHONPATH` export; isolation there rests entirely on `849e3ca`'s
   `SBATCH_EXPORT=ALL` propagating the submitting shell's environment through
   `#!/bin/bash -l` profile sourcing, `module load cuda/11.4`, and SLURM's own
   export handling. That is a single point of failure with no in-job fallback,
   and per the project record it had never been exercised by a live job before
   this experiment's training run — which did work. `scripts/manager/run_training.sh`
   has the same gap on both trees and is a live trap for the next manager-slot
   experiment. For the maintainer: the union template should be ported back.
9. **Training (step 6), SLURM job 29870374, 00:04:17 on one A100.** 1.21x the
   base run's 3m32s, comfortably inside §5's 3x rule. Provenance verified from
   inside the job's own log two independent ways: a warning traceback naming
   `/u/certuer/autoresearch/switch-joint-exodus/src/aimanager/generic/data.py`
   (the string `algorithmic-institutions/src` appears zero times in the log), and
   the branch-only per-fold joint-loss lines, which the shared tree cannot
   produce because `aimanager.generic.joint_exodus` does not exist on `main`.
   Artifact sha256 `8a4ae4ade60d5443970255a4265bb7abaf555164ec020a4c77ba28ff364abdc0`,
   identical in-job and after fetch, carrying `joint_exodus = True`,
   `joint_exodus_switch_every = 4`, an inert copula (`copula_rho = 0.0`), and a
   1,131-parameter head. The joint loss descended monotonically in every fold,
   ~3.00 -> 1.92-2.12, still falling at epoch 374 in all five.
10. **The detach held on real hardware.** Per-agent held-out log-loss came out at
   0.5169944527434331 against the base artifact's 0.5163464160282509 — a
   difference of +0.00065 with *mixed signs*, four of five folds improving. The
   attached head in step 2b cost +0.0042 systematically; this is 6.5x smaller and
   non-directional, which is the signature of a different random realisation
   (constructing the head consumes RNG) rather than a degraded model. The trunk's
   objective is unchanged by construction, and the gradient-identity test passes.
11. **The local PyG stand-ins do not diverge from real PyG.** All 87 of this
   experiment's tests pass on Raven against real `torch_scatter` 2.0.9 and
   `torch_geometric` 2.5.0, and each test module was made to report whether its
   stand-ins were installed — every one reported not-installed, confirming the
   real packages were exercised. No local result in this experiment needs
   re-examining. The 10 non-passing tests in that run are eval-suite and
   linear-manager tests missing fixtures under `artifacts/` and `plots/`, which
   `remote_test.sh` excludes from its sync by design and which CLAUDE.md says run
   locally.
12. **Verdict: `[FAIL]`.** Gate 1 passes decisively — SC 2.1867905905576634 ->
   1.147392663266986, a band upgrade 2-5 -> 1-2, the largest single-row
   improvement this slot has recorded on this stack. Gate 2 fails: the 21-row
   mean rose from 1.2893632310269196 to 1.3040409569053069, +0.0146777258783870.
   Both gates are required (§2), so this is a fail whatever the size of the SC
   move. For the maintainer's calibration at review, the mean margin sits between
   the two standing precedents: PR #164 was overridden at 0.003, PR #168 was not
   at 0.031. This one is 0.0147. I am not arguing the override; I am recording
   where it falls.
13. **SC did not get fixed — its error changed sign.** The larger-group-size
   distribution now *overshoots* the human right tail. Mass on the fully merged
   state: human 0.1440, parent 0.0560, candidate 0.2140 — the parent's shortfall
   of -0.0880 became an excess of +0.0700, and every other bin is correspondingly
   deficient. Mean larger-group size: human 6.088, parent 5.676, candidate 6.286.
   The EMD fell from 0.4119999999999999 to 0.19799999999999995 against a ceiling
   of 0.19452799999999945, which is why the score lands just above 1. The residual
   is now entirely "too much merging", where the parent's was "too little", and
   the per-round line turns *upward* over rounds 16-20 (6.13 -> 6.33) where the
   human line declines (5.92 -> 5.84). The trajectory shape is wrong at the end of
   the game, not merely the level. A successor should aim at calibrating the dose,
   not at adding more.
14. **The conditional-Bernoulli protection was falsified, and the reason is
   structural.** The Declaration asserted that conditioning on the count and
   selecting movers by propensity would protect RCD, the row PR #168's shared
   latent broke. RCD degraded anyway, a full band, 1.9647336396755046 ->
   2.764919035295771 — the largest movement in the table. The protection is
   vacuous exactly where the mechanism is most active: when the head draws
   m = k, there is precisely one subset (C(k,k) = 1) and no selection is made at
   all — everybody leaves regardless of propensity. Measured on the two runs'
   own parquets, full-exodus cells rose from 0.0869 to 0.1424 of decision cells
   (human reference 0.1079 from step 2's clean estimate, so the candidate
   overshoots the human rate too), and the share of all movers sitting inside a
   full-exodus cell rose from 0.1766 to 0.3043. Nearly a third of the candidate's
   movers are selected by no propensity whatsoever. **Any successor that buys SC
   through full-group exodus inherits this: the assortativity defence and the SC
   lever are in direct tension, because the same event supplies both.**
15. **The collateral splits cleanly along one seam.** Every contribution row
   improved (CA -0.123, CB -0.114, CD -0.112, CE -0.205, CC -0.044, CF -0.075
   crossing into the ceiling) and the punishment family improved (PA, PB, and PD
   -0.057 crossing into the ceiling), giving three band upgrades in total with SC.
   Every consequential worsening is in the relational/switching block: RCD +0.800
   and RCB +0.286 (both 1-2 -> 2-5), RSA +0.421 and SB +0.206 (both leaving the
   ceiling), with RCA and RCC drifting up within band. The improvements sum to
   -0.7735 and the worsenings to +0.9818, which is the whole of gate 2's failure.
   Three upgrades against four degradations, rows <= 1 unchanged at 11/21 with the
   composition rotated: CF and PD in, SB and RSA out.
16. **SB behaved exactly as PR #169 note 11 predicted**, which is worth recording
   as a successful forecast rather than a surprise: a mechanism that drives the
   simulation into more unbalanced states raises the realised switch rate in the
   late decision rounds, and SB pays. SB 0.8592497281751998 -> 1.06526788518747,
   raw mean per-round gap 0.0166 -> 0.0263 against a ceiling of 0.0394. SA stayed
   inside the ceiling but its raw gap grew 4.8x, 0.00208 -> 0.00992.
17. A measurement convention worth fixing in memory, because it nearly misled the
   reading of this run. SC is defined over rounds 4 onward
   (`metrics.py`, and `evaluation_metric_defs.md`: rounds 0-3 excluded because
   groups are always 4-4 before the first switch). Computing the same statistic
   over all 24 rounds drags the parent's mean larger-group size from 5.676 to
   5.3967, purely by adding 400 rounds that are 4 by construction. Restricting
   instead to the five post-decision anchor rounds {4, 8, 12, 16, 20} gives a
   bit-identical mean to the full window, since membership is constant between
   decision rounds — so the anchor convention can never explain a mean
   discrepancy, only a change in n. Mixing the two conventions across a human and
   a simulated number makes an overshoot look half its true size.
18. **Maintainer ruling, 2026-09-03: this is a SUCCESS.** The maintainer deemed
   the result a success and stated the intention to amend the §2 definition
   accordingly. The measured numbers are unchanged and are not restated here in
   any softer form: SC 2.1867905905576634 -> 1.147392663266986 (band upgrade),
   21-row mean 1.2893632310269196 -> 1.3040409569053069 (+0.0146777258783870,
   i.e. gate 2 as written at the time of the run was not met). What changed is
   the adjudication, not the measurement. Recorded here in the form PR #150 used
   for its own maintainer ruling, so that a later reader can see exactly which
   part is data and which part is judgement.
19. **Flagged to the maintainer before any §2 edit: the definition change is
   retroactive.** Gate 2 is what separates a band upgrade from a band upgrade
   that was paid for elsewhere, so relaxing or removing it reclassifies past
   experiments as well as this one. On the record as it stands, at least these
   would become successes: PR #168 (all four arms band-upgraded SC, none passed
   the mean), PR #163 (RCD band upgrade with an RCA collapse — whose entire
   scheduled-sampling family the maintainer vetoed on 2026-08-27), PR #154
   (CG and RCD upgrades, killed by the marginal C block), PR #156 (a two-band
   RCD upgrade, killed by RCA), and PR #151 (RCA upgrade, killed by the CG
   explosion). PR #169 and PR #162 would remain failures, having produced no band
   upgrade at all. The narrower instrument with existing precedent is a per-PR
   maintainer override recorded in the title, as PR #164 did on a 0.003 margin;
   it reaches this result without reopening the other five. Note also that a
   "net band movement" rule would not rescue this experiment either — it posts 3
   upgrades against 4 degradations.
20. **Superseding notes 12 and 18: this passes both gates as written, and needs
   no override.** `main` commit `b174f90` (2026-09-03) amends §2's second gate
   from "the mean improves" to "the mean may not rise more than 10% above the
   evaluation stack's baseline mean". Against the baseline 1.2893632310269196
   that ceiling is 1.4182995541296117. The candidate's 1.3040409569053069 is a
   rise of **1.1384%**, leaving 0.1142585972243049 of headroom — comfortably
   inside. Gate 1 was never in question. So the `[SUCCESS]` title now rests on
   the criteria themselves rather than on a maintainer override, which is a
   stronger footing; note 18's ruling stands as the record of how the
   adjudication got here. No measurement changed at any point.
21. **The retroactive scope of `b174f90`, narrowed by the actual rule chosen.**
   My earlier flag assumed gate 2 might be removed outright, which would have
   reclassified five prior experiments. A 10% margin is far more conservative,
   but it is not inert: PR #168's best arm (mean 1.3208 against the same
   1.2894 baseline, and a two-band SC upgrade to 0.9588) sits inside the new
   ceiling too and would now qualify. Whether that PR is revisited is the
   maintainer's call; recording it so the next reader is not surprised.
22. **Merged `origin/main` into this branch (`dade303`) rather than rebasing onto
   it, deliberately.** The evaluation stack for this experiment is PR #165's, and
   `main` contains neither of the artifacts the simulation config names
   (`group_switching_contribution_50ep_herding_copula_v2`,
   `punishment_multinomial_severity_copula.joblib`) nor any copula code at all —
   `git show origin/main:src/aimanager/generic/graph.py | grep -c copula` returns
   **0**. Rebasing onto `main` would therefore not carry this result with it; it
   would silently re-target the candidate at a different stack, and §3's baseline
   and both gate figures would cease to apply. Merging keeps the parent stack
   intact while bringing main's tooling and the amended §2 onto the branch.
23. **The merge conflicted in exactly the predicted place, and confirms `main`
   still has the gap.** Both SLURM templates conflicted — our in-job
   `export PYTHONPATH="$PWD/src..."` against `main`'s *absence* of it. Resolved to
   the union again. So as of `b174f90`, `main`'s isolation still rests entirely on
   `SBATCH_EXPORT=ALL` propagating `PYTHONPATH` from the submitting shell
   (`train_cluster.sh:107-111`), with no in-job fallback.
