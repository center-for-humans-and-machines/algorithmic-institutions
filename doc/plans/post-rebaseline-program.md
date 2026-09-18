# [ACTIVE] Post-rebaseline program: four steps

## Context

Four investigations and one re-baseline established a sharp picture of what is left wrong with the artificial-human stack, and it is narrower than it looked.

The contributor's one-step conditional is correct: retrained with each of five folds held out, its teacher-forced punishment-response statistic is 0.095 against 0.082 in-sample and a noise ceiling of 0.348 (PR #183). Its per-round noise magnitude is correct: predictive SD about 3.1 and residual variance 11.2-11.8 against the human 11.8 (PR #186). The within-group residual correlation is real, small and correctly dosed: human 0.032, the calibrated sampler 0.031 (PR #186, PR #187). Parameter uncertainty is not a material source of shared error: a five-seed ensemble implies a correlation strength of 0.005-0.009 against the fitted 0.0395, with the wrong time structure, and replacing the sampler with one ensemble member per episode scores like having no sampler at all (PR #188). A static per-group effect does not exist in the human data: the residual dependence is a round-local shock with a two-thirds echo into the next round and nothing beyond, and the static share's interval runs from -0.33 to +0.01 (PR #187).

What remains is one defect. The deterministic map contracts when it is iterated off the human trajectories. With the sampler disabled, the variance of the model's own conditional expectation over the states it visits is 18.9 against 27.9 on human histories, while its residual variance is unchanged. The group-spread row, the switching-pull row and the attenuated closed-loop punishment response are all symptoms of that one thing, and the sampler's episode-long persistence has been masking it.

The punisher lag fix (PR #184) moved the manager-policy row from the 1.23-1.56 range into 0.68-0.89 in all six rerun stacks and the binned reaction row down by 0.5 to 0.9 everywhere, but left the reaction at the ceiling untouched, because the retrained punishers still punish full contributors three to four times too often.

## The four steps

### Step 1 -- Settle the emission-head question before building anything

**Question.** Does a location-scale emission head resist off-manifold contraction better than a categorical head over 21 free logits? This is the main thing a combined "all-star" architecture would take from the Gaussian-MLP lineage, and it is currently unresolved.

**Reasoning.** A categorical head has no structure tying the 21 levels together, so away from the training manifold it can relax toward the training marginal. A location-scale head moves a scalar and keeps extrapolating a monotone shift. The same property would explain both the Gaussian lineage's better response slopes and, if it holds, a smaller share of the closed-loop contraction. Counter-evidence to respect: the Gaussian stacks are not better on the group-spread row (1.85 and 1.67 against 1.55), though their switch models differ, so the comparison is confounded.

**Method.** No training and no new models. Measure the state-spread decomposition of PR #186 on the two Gaussian stacks' existing post-fix simulations, plus copula-disabled counterparts so the comparison is like for like against the categorical trunk's 18.9. Add an off-manifold gain probe: push the group's recent contribution level away from the human states by 2, 4 and 6 points and record how much each model's predicted expectation still moves.

**Branch.** `auto/head-state-spread-diagnostic`. **Cost.** Two short simulations, no training.

**Done when** the log states whether the location-scale head holds its state spread and its off-manifold gain better, by how much, and whether that supports building the combined head.

### Step 2 -- Fix the manager at the contribution ceiling

**Question.** Can the reaction-at-the-ceiling row be moved by letting the punisher express a discontinuity at the maximum contribution?

**Reasoning.** Real managers punish a full contributor 4% of the time and, when they do, punish hard (mean 7.0). The retrained linear manager punishes 13-14% of the time at a mean of about 4, the graph-network manager 11-14% at about 5. Both encode contribution as a single numeric feature, so neither can express a sharp break at exactly 20; they interpolate the ceiling from the 15-19 band. The reaction-at-the-ceiling row is defined on punished full contributors, so the simulation is fabricating its population. This is the only row the re-baseline did not move.

**Method.** Confirm the diagnosis on the human data first. Then add an explicit gave-the-maximum indicator (and consider gave-nothing) to both punisher families, retrain, and carry the severity copula strength over unchanged. Rerun the frontier stack and the graph-network-manager reference and evaluate all 22 rows.

**Branch.** `auto/punisher-ceiling-fix`. **Cost.** One linear retrain (about a minute), one graph retrain (about eight minutes), two simulations.

**Done when** the reaction-at-the-ceiling row has a declared verdict under the two gates with the protected response row intact, and the mechanism table shows the two probabilities and the severity profile against the human row.

### Step 3 -- Port the one-hot group-size switch head onto the frontier trunk

**Question.** Do the Gaussian lineage's better switching numbers belong to its switch head or to its contributor?

**Reasoning.** The switch slot is largely separable from the contributor, so this is a cheap partial merge of the two lineages that does not depend on step 1's outcome. The stack carrying the one-hot joint exodus head posts the best segregation row (1.076), the best switching-pull row (0.760) and the best response slope (0.705) in the set, while its weaknesses sit in the contributor.

**Method.** Establish whether the code path and the trained artifact already exist on the frontier lineage; the joint exodus head should be in its ancestry and the one-hot artifact was copied across during the re-baseline. Swap only the switch model in the frontier simulation config, simulate, evaluate.

**Branch.** `auto/switch-kexo-port`. **Cost.** One simulation; a retrain only if the switch model's inputs turn out to depend on the contributor.

**Done when** the 22-row comparison against the frontier baseline is recorded with a verdict, and the log says what the result implies about the attribution question.

### Step 4 -- Freeze the noise model and change how trunk changes are judged

**Question.** Not an experiment. A protocol change so that steps 1 to 3 and everything after them are attributable.

**Reasoning.** Today a copula recalibration rides along with every trunk change, so a contributor experiment and a noise-model experiment move at once and cannot be told apart. Separately, judging contributor changes on the group-spread row is misleading while the sampler's persistence is supplying most of that row: PR #186 showed the persistence contributes a factor of 5.3 by compounding, and the state absorbs it by the late rounds.

**Change.** The correlation strength and persistence are frozen per model family and added to the frozen surface; altering either is its own declared experiment. Contributor-trunk changes are judged with the sampler disabled, on the state-spread diagnostic against the human 27.9, alongside the usual gates.

**Deliberately not done yet.** PR #187 showed the human dependence has the shape of a one-round echo rather than the episode-long persistence that is shipped. Changing the shape now would cost the group-spread row and the high-contributor withdrawal slope with nothing in place to replace the variance they currently borrow from it. Keep the shipped shape, stop letting it move, and measure past it. Revisit once a trunk change closes the state-spread gap.

**Done when** the protocol records the freeze and the diagnostic, and this plan is committed.

## Sequencing

Steps 1, 2 and 3 are independent and run in parallel, each on its own branch against `auto/punisher-current-contribution` with its own isolated cluster directory. Step 4 lands alongside them and governs how 2 and 3 are judged.

Step 1 gates the decision that follows this program. If the location-scale head holds up, the next experiment is a combined head on the frontier trunk: the graph body with its per-group node and direct punishment path, and an ordinal location-scale emission with explicit inflation at both corners and at repeat-previous, which is designed to keep the categorical head's corner behaviour while gaining the monotone-shift structure. If it does not hold up, the combined architecture reduces to whatever step 3 shows, and the contraction has to be attacked another way, most plausibly through the gated version of the direct punishment path.

## Explicitly out of scope

The three group-state features from PR #187 (share of group peers punished last round, group contribution trend, within-group spread) are worth folding into the next contributor retrain, but they account for about a seventh of a small correlation and do not justify their own cycle.

The rollout-training family is the wrong tool for this defect. The one attempt (experiment #163) fixed the switching-pull row dose-responsively but left the group-spread row in its worst band, and damaged the punishment response because substituting the model's own contribution while keeping the human punishment pairs a punishment with a contribution that did not earn it. That mechanism is now understood and fixable, and the curriculum code is committed and tested, so the family may deserve a second look later; it is not the answer to the contraction.

## Open after this program

The mid-range response band (contributions 10-14) has the wrong sign in every condition, including teacher-forced and held out, so no closed-loop fix will supply it.

The manager's action support: humans rarely punished above 10 or punished high contributors, so a reinforcement-learning manager exploring there operates on roughly 300 rows of evidence. Either bound it near the human range or audit a trained policy for how often it leaves it.

The lineage merge itself, once step 1 and step 3 have reported.

The 32-stack sweep matrix has not been rerun under the fixed punisher, so the ledger's deficit profiles are pre-fix.
