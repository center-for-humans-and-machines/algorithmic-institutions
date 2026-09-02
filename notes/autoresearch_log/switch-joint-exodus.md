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
