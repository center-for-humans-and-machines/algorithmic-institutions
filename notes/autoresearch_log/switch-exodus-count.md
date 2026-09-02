# Collective exodus: a learned group-level count head for the switch slot

## Declaration

**Slot:** switch.

**Parent:** PR #165 (`auto/contribution-herding-copula-v2`), the maintainer-designated
frontier. Branch and worktree created from `origin/auto/contribution-herding-copula-v2`
at `4be127a`; the PR opens with `--base auto/contribution-herding-copula-v2`.

**Base model:** the GNN switch predictor
`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`
(`x_encoding = common_good, punishment, agent_group, round_number`;
`edge_encoding = []`; `y_levels = 2`). It carries no copula fields, so it runs
the legacy independent `th.multinomial` draw.

**Evaluation stack (§3, parent rule §9):** the parent's own stack and config,
`23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch`, single
pairing `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
`save_per_round: true`.

**Baseline (the parent's confirmed scores — both §2 gates are judged against these):**

| quantity | value |
|---|---|
| SC | 2.1867905905576634 (band 2-5) |
| mean over 21 rows | 1.2893632310269196 |
| rows <= 1 | 11/21 |

**Target row:** SC, 2.1867905905576634, band 2-5 -> requires 1-2 or better.
**Guard (gate 2):** the 21-row mean must fall below 1.2893632310269196.

### Hypothesis

A minority group leaves *together*. When a human sits in the smaller group, the
decision to abandon it is a collective one — nobody wants to be the one left
behind — so the number of leavers from a small group is far more concentrated on
"all of us" and "none of us" than independent decisions allow. The effect is a
property of **group size**: it is strong in a group of 3-4, weak from 5 up, and
absent once everyone has merged into a group of 8, where there is no minority and
nobody to be abandoned by.

Measured on the human data (single copy per game, 50 games, 1,891 valid
decisions), the count of leavers per (episode, decision round, group) is
over-dispersed relative to binomial by 2.27x at size 4 and 1.90x at size 3,
against 1.28x/1.72x/1.92x at sizes 5/6/7 and 0.81x at size 8. A minority group of
4 leaves unanimously in 21/120 = 17.5% of cases where independence predicts 2.7%
(6.4x); a group of 3 in 13/67 = 19.4% against 5.8% (3.3x). The simulation
reproduces roughly half of this exactly where it matters (method-of-moments
rho-hat: size 3, 0.449 human vs 0.244 sim; size 4, 0.422 vs 0.247) and matches
humans at sizes 5-8. Recomputing each variance ratio around each game's own
switch rate leaves it unchanged (size 3: 2.25 -> 2.26; size 4: 2.83 -> 2.87), so
this is genuine within-round coordination, not game-level heterogeneity.

That deficit is the whole SC row. SC's gap is a missing right tail — the sim
reaches a fully merged group 5.6% of observations against the human 14.4%, mean
larger-group size 5.676 vs 6.088 — and the human signature for arriving there is
the simultaneous emptying of a minority: humans go from a 5-3 split to a merged 8
in 8 of 51 transitions (15.7%) where independence predicts 0.4%; the sim manages
3.8%. Decomposing the 0.412 gap in mean larger-group size against
independent-null Monte Carlos built from each source's own rates, about 0.13 is
marginal switch rates and about 0.28 is correlation strength.

**Planned change.** Factor the group's joint switch decision into "how many leave"
and "which of them leave", and learn the first factor instead of assuming it.
A group-level **count head** is added to the switch GNN: the node embeddings of a
group's members are pooled (masked by `agent_group`) and read out as a categorical
distribution over the number of leavers m = 0..k, conditioned on the group's state
including its size k and the round. At simulation time each group draws m from
that head and then draws *which* m members leave from the **conditional Bernoulli**
distribution induced by the existing per-agent switch probabilities — exact by
enumeration, since a group of k <= 8 has at most C(8,4) = 70 subsets. The existing
per-agent head is untouched and keeps deciding who.

**Why this and not more correlated sampling.** Within-round unanimity that exceeds
what shared observables explain is residual dependence by definition, so a joint
sampler is forced; what is *not* forced is the Gaussian-copula parameterisation the
slot has used five times. That family failed for reasons this decomposition
removes by construction:

- *It picked the wrong movers.* A shared latent selects leavers partly by group
  draw rather than individual propensity, so movers arrive mismatched
  (assortativity 0.278 human vs 0.110 under the copula) and RCD band-degraded in
  all four arms of PR #168's grid. Conditional Bernoulli conditions on the count
  and leaves the propensity ordering intact, so *who* moves is still the trained
  model's answer.
- *It dosed uniformly.* A constant rho over all cells and all five decision rounds
  necessarily overdoses sizes 5-8 and rounds 7-19, where the parent already matches
  the human co-switching (r 0.057/0.049/0.072/0.108) — which is where the C-family
  drift that cost PR #168 the mean gate accrues. A head conditioned on size and
  round learns to add dispersion only where the data shows it.
- *It was a hand-calibrated knob.* The count head is fitted by maximum likelihood
  on the observed counts, and being categorical over m it can represent the
  unanimity spike at m = k, which a single correlation parameter cannot.

**Rows this should move, and how.** SC, by restoring the minority-exodus transitions
that build the missing right tail. SA and SB are the declared watch items: the
head fixes E[m] directly, so the overall and per-opportunity switch rates should
hold, and they are the first thing to check. RCD is the row PR #168 broke and this
design is meant to protect.

**Falsified before starting, and recorded so it is not retried.** The experiment
opened on a different hypothesis — that the sim's excess restoring drift comes
from a size-blind per-capita leave rate, and that giving the model group size
would fix it. It is dead on two independent counts. (1) The model is not size-blind:
the graph is complete over all 8 agents, so degree is pinned at 7 and headcount at
8, making `scatter_mean` an exact affine function of own-group size
(`agg_i = m_other + ((n_own-1)/7)(m_own - m_other)`) — size is linearly decodable
from the first message-passing layer. (2) The sim already reproduces the human size
dependence: logistic regression of `does_switch` on own-group size with round fixed
effects and game-clustered SEs gives -0.1481 (se 0.0382) human against -0.1432
(se 0.0272) sim, a difference of -0.0048 (se 0.0470, z = -0.10). A Monte Carlo of
independent switching at human size-conditioned rates lands at mean larger-group
size 5.20-5.37 across every smoothing choice — *below* the 5.676 the simulation
already achieves, and far below the human 6.088. Size dependence is real, small,
and already present.

## Plan

Validated by the orchestrator against §2 (targets), §5 (legality) and §8 (frozen
surface) before any step ran. No step adds an input feature, so the model
consumes exactly the information it consumes today and the
`Environment.default_values` hazard of note 3(b) cannot bite; the only new
learned object is a factorisation of the *label* distribution, which is a
likelihood choice, not a feature. Nothing under
`src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
`notes/eval_scoring_schema.md` or `experiments/` is touched, and the simulation
protocol, seeds and episode count are the parent's.

1. **Port the isolated-run launcher fix** — `scripts/run_simulation.sh`.
   Apply commit `66733b3` from `auto/switch-herding-copula-recal`: export
   `PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"` after venv activation, so a
   job launched from `~/autoresearch/switch-exodus-count` imports this branch's
   `src/` instead of the shared checkout's. Nothing on this branch may be
   simulated before this lands. *[Sonnet]*

2. **Preflight gate — does the observed count distribution reach the target?**
   New scratch analysis (not committed to `src/`), on the single-copy human file
   and the parent's `per_round.parquet`. Estimate the empirical leaver-count
   distribution p(m | k, round) on human decisions, then run the group-size Monte
   Carlo of the diagnostic drawing m from it directly, with membership assigned by
   propensity. Report the resulting larger-group-size distribution and mean against
   the human 9.6/24.4/28.0/23.6/14.4 and 6.088, plus the unanimity rates at k = 3
   and 4. **STOP condition:** if drawing from the human count distribution itself
   does not lift the mean to within roughly 0.1 of 6.088, the factorisation cannot
   carry SC and the experiment stops here and is written up as a `[FAIL]` with the
   ceiling recorded. Also report E[m] against the sum of the base model's per-agent
   probabilities on the same rows — the SA/SB consistency check. *[Opus]*

3. **The count head** — `src/aimanager/generic/graph.py`, `GraphNetwork`.
   Add an optional group-level head: pool the post-RNN node embeddings over each
   group (mask by the `agent_group` column of the state, not by the edge index,
   which is group-agnostic), concatenate the group size k and the round, and read
   out 9 logits for m = 0..8, masked to m <= k before the softmax. Gate the whole
   head behind a constructor argument defaulting to off, and persist it through
   `save()`/`load()` alongside the existing `copula_*` fields so an artifact
   without it loads exactly as today. *[Opus]*

4. **The joint training objective** — `src/aimanager/artificial_humans/train.py`
   plus a new config
   `configs/training/artificial_humans/switch_predictor/exodus_count.yml`.
   Derive the per-(episode, decision round, group) leaver count from the existing
   `does_switch` labels under the `switch_valid` mask, and add its cross-entropy
   to the existing per-agent loss. The config copies
   `opt_50ep_doubled_reanchored.yml` unchanged (375 epochs, batch 10, lr 5e-4,
   hidden 10, 5-fold, seed 38381, flip-doubled data per the GNN convention) and
   only enables the head. Report both loss components per fold. *[Opus]*

5. **The conditional-Bernoulli sampler** — new
   `src/aimanager/generic/conditional_bernoulli.py`, with unit tests that run
   locally (no PyG import). Given a group's per-agent probabilities and a count m,
   draw the leaving subset exactly by enumerating the C(k,m) <= 70 subsets with
   weight proportional to the product of odds. Tests must assert: the marginals
   recovered by averaging over m drawn from a binomial match the input
   probabilities; the propensity ordering is preserved (a higher-p agent leaves at
   least as often as a lower-p one); and m = 0 and m = k are handled. *[Sonnet]*

6. **Wire the sampler into the simulation draw** — `graph.py`, alongside the
   existing copula dispatch in `predict_independent`. When the count head is
   present and `sample=True`, replace the independent `th.multinomial` draw on
   decision rounds with: per-agent probabilities from the existing head, m drawn
   per group cell (`batch_index * 2 + agent_group`) from the count head, then the
   step-5 subset draw. Non-decision rounds keep running the forward pass to hold
   the GRU warm and consume no extra RNG, matching the current semantics. *[Opus]*

7. **Train/sim parity test** — `src/aimanager/tests/`. Assert that the grouping
   and count derived at training time from `data.py` matches the grouping the
   simulation builds from `environment.py` on the same synthetic membership,
   closing the untested invariant of note 3(a) for this mechanism. *[Sonnet]*

8. **Train on Raven** — `scripts/train_cluster.sh ah <config>` with
   `AI_REMOTE_DIR='~/autoresearch/switch-exodus-count'`. Base run is 3m32s, so the
   §5 budget ceiling of ~10.5 min is not in question; record the actual elapsed
   time and the in-job artifact sha256. *[Sonnet]*

9. **Baseline control, then the candidate** — two simulations from the isolated
   dir. First re-run the parent's own config unchanged and require a bit-identical
   `per_round.parquet` (sha256 `4f64fc42...` per PR #168's control), which is the
   licence to compare anything; then the candidate config, a three-edit copy of the
   parent's differing only in `switch_model`, `output_dir` and `figure_name`.
   *[Sonnet]*

10. **Evaluate and rule** — `python -m aimanager evaluate` on the candidate,
    locally. One simulation, one evaluation, no second stage (§3). Record the
    results row, then open the PR titled `[SUCCESS]` only if SC leaves band 2-5
    *and* the mean falls below 1.2893632310269196; otherwise `[FAIL]`. *[Opus]*

## Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## Notes

1. The preflight was run before any code was written, on the parent's own
   simulation output and the single-copy human file, and it killed the opening
   hypothesis outright (see the Declaration's final paragraph). Keeping the
   dead hypothesis on the record is the point: "add group size to the switch
   model" is now closed, with the reason quantified, and the diagnostic that
   closed it is what produced the count-head hypothesis.
2. Absorption of the merged state needs no mechanism and must not get one.
   Humans hold a size-8 group at 0.300 (9/30 transitions), and pure independence
   at the human size-8 leave rate of 0.147 predicts (1-0.147)^8 = 0.288. The
   human rho-hat at size 8 is indistinguishable from zero. The SC failure is
   entirely about *entering* the merged state, never about holding it — any
   mechanism that makes size 8 sticky is fixing a problem that does not exist.
3. Two hazards in the plumbing, found in the code map and to be designed around
   rather than discovered at run time. (a) Train-time node features are built in
   pandas (`src/aimanager/generic/data.py`) and sim-time features in torch
   (`src/aimanager/manager/environment.py`) — two independent implementations
   with no shared path and no test asserting they agree; the existing precedent
   `own_grp_prev_mean_contr` keeps them in sync by comment only. (b)
   `Environment.default_values` is taken from the *contribution* artifact, not
   the switch artifact (`environment.py:67-71`), so a new key known only to the
   switch model is silently dropped from `reset_state`'s `prev_` loop and
   `KeyError`s at simulation time, after training has already succeeded.
4. `scripts/run_simulation.sh` on this parent still carries the PYTHONPATH bug
   that voided PR #166: an isolated `AI_REMOTE_DIR` borrows the canonical venv,
   whose editable install resolves `aimanager` to the *shared* checkout, so the
   branch's own `src/` is silently ignored. The fix exists as commit `66733b3`
   on `auto/switch-herding-copula-recal` and must be in place, with a
   bit-identical baseline control run, before any score from this branch is
   believed.
5. Method-of-moments rho-hat understates the dispersion that the SC row actually
   needs: plugging the measured per-size rho into an exchangeable-correlation
   Monte Carlo yields mean larger-group size 5.779 against the human 6.088,
   recovering +0.41 of the +0.72 the human data shows over its own independent
   null. This is an argument for fitting the count distribution directly by
   likelihood rather than through a correlation parameter, and a warning that
   the head must be checked against the observed unanimity rates (17.5% at
   size 4, 19.4% at size 3), not against a summary statistic.
