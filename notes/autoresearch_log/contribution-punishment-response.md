# Sharpening the contributor's response to punishment (RCB)

## 1. Declaration

**Slot:** contribution.

**Parent:** PR #179 (`auto/contribution-group-vnode`, `[SUCCESS]`), the
maintainer-designated frontier, at `11fe223`. Branch
`auto/contribution-punishment-response`, worktree
`.claude/worktrees/contribution-punish-onehot`, created from
`origin/auto/contribution-group-vnode`; the PR opens with
`--base auto/contribution-group-vnode`.

**Base model:** the parent's contributor -- the M0 GNN trunk plus the per-group
virtual node, retrained (`group_switching_contribution_50ep_group_vnode`), with
the copula recalibrated on it (`rho = 0.0435568043640977`, `phi = 1.0`,
`switch_every = 1`), shipped as
`artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`.
`x_encoding = prev_contribution (numeric, 21), prev_punishment (numeric, 31),
agent_group (onehot, 2)`; hidden 20; `add_global_model: False`;
`group_vnode: True`; 575 epochs, batch 4, lr 3e-4, seed 38381, flip-doubled data.

**Evaluation stack (§3 under the parent rule of §9):** the parent's own config
`configs/simulation/manager_testing/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch.yml`
-- this contributor x the joint-exodus GNN switch predictor x the severity-copula
`lin_multinomial` punisher, single pairing `lin_multinomial_copula_self`, seed 42,
100 episodes, 24 rounds, `save_per_round: true`.

**Baseline (the parent's confirmed scores; both §2 gates are judged against these).**
Source: `plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch/evaluation/scores.csv`
in this worktree.

| row | score | band |
|---|---|---|
| **RCB** (the only target) | **2.3151705700149083** | 2-5 |
| RCC | 1.6596292609904482 | 1-2 |
| RCA | 1.4000972794261808 | 1-2 |
| RSA | 1.3546243055859613 | 1-2 |
| RCD | 1.3404829181218743 | 1-2 |
| RPA | 1.3111568561676463 | 1-2 |
| SB | 1.110991519531801 | 1-2 |
| CE | 1.1106638835612843 | 1-2 |
| CF | 1.0762250158942481 | 1-2 |
| mean over 21 rows | **1.0988293946890038** | |
| gate-2 ceiling (mean x 1.10) | **1.2087123341579042** | |
| rows <= 1 | 12/21 (context only) | |

**Target row.** **RCB alone.** It is the only row on this parent still at or
above 2, so §6's "rank your rows with score >= 2" returns a list of one. Band
2-5 -> 1-2 requires the resampled per-bin discrepancy below
**0.6957104109931878** (2 x the noise ceiling 0.3478552054965939) against the
parent's **0.8053441343922023** -- a **13.6%** reduction. On the raw canonical
frames the same statistic is 0.7973084747030883.

### Hypothesis

**The behaviour.** *The same punishment means opposite things depending on what
you gave.* Punished after contributing little, a player complies and raises;
punished after contributing a lot, a player reads it as unfair and withdraws.
Measured on the human canonical frames over the RCB population (punished,
non-full contributors, n = 2,660), the slope of the next-round contribution
change on the punishment received, taken **within** contribution band:

| own contribution | human slope of dc on p | parent sim | sim / human |
|---|---|---|---|
| 0-4 | **+0.1397** | +0.0619 | 44% |
| 5-9 | **+0.1038** | +0.0115 | 11% |
| 10-14 | **-0.0767** | -0.0082 | 11% |
| 15-19 | **-0.1615** | -0.0366 | 23% |

The sim has the **right sign pattern everywhere and a quarter of the
magnitude**. The compliance response, the sign flip at the midpoint, and the
withdrawal response are all present and all attenuated. The cell means say the
same thing: humans move +1.15 -> +3.96 across punishment size at contribution
0-4 and -1.06 -> -4.56 at 15-19; the sim moves +1.18 -> +2.41 and -1.44 -> -2.11.

**It is not the ratio.** The metric bins by `punishment / (20 - contribution)`,
but that ratio is not the behavioural variable: regressing human dc on
contribution, punishment and the rate together leaves the rate with coefficient
**-0.6871** -- the wrong sign -- while punishment carries **+0.1337**. The top
rate bin's negative sim mean (-0.151) is a *composition* fact, not a sign error:
that bin is dominated by high-contribution rows, where humans also go negative.
A feature built on the ratio would be engineered at the metric's definition
(§5, illegal) *and* aimed at the wrong construct.

**Fixing the conditional response is sufficient.** Decomposing the scored
statistic over the (contribution, punishment) cells:

| | weighted discrepancy | band? |
|---|---|---|
| parent sim, as scored | 0.7973 | 2-5 |
| sim's conditional response @ human cell composition | 0.5707 | 1-2 |
| **human conditional response @ the sim's own cell composition** | **0.3731** | **1-2** |

The sim's cell composition is not what keeps RCB out of the band: holding it
fixed and giving the model the human response alone takes the statistic to less
than half the band edge. The lever is the conditional, and roughly half of the
attenuation needs to be closed, not all of it.

**What the change will be, and the rule that picks it (stated before its numbers
are visible).** Two mechanisms can attenuate a response the model demonstrably
*can* learn, and they call for different changes. A teacher-forced surrogate of
the per-agent path -- one `Linear(., 20) + Tanh` over the lagged own pair, then
the 21-way readout, the same width and the same single nonlinearity the trunk
has -- reproduces the human slopes from **numeric** inputs alone
(+0.152 / +0.106 / -0.061 / -0.233 against human +0.140 / +0.104 / -0.077 /
-0.162), and keeps doing so when the trunk's real competitors (own lags 2 and 3,
the own-group peer mean) are added. So a flat first layer is *not* provably the
binding constraint, and the experiment must not assume it is. Step 0 therefore
measures the parent trunk's own **teacher-forced** within-band slopes on the
human data, and the change follows this rule:

- **(a) The trunk's teacher-forced slopes are already attenuated** (the 0-4 band
  below +0.10, or two of the four bands below half the human value). The
  conditional itself is underfit, and the lever is the representation of the
  lagged own pair: **one-hot `prev_punishment`**, and `prev_contribution` with
  it if step 0's ratio is worse than 0.6 in the two bands where the sign flips.
  Rationale: "not punished at all" is 68% of the rows and is a different event
  from "punished one point", and a numeric scalar makes them one step on a
  single line; the same ordering constraint is why M0 already one-hots
  `agent_group` and why PR #174 one-hot the joint head's group sizes. In the
  surrogate this cuts the RCB bin gap 0.2226 -> 0.1638 with numeric
  contribution and -> 0.1433 with both one-hot.
- **(b) The trunk's teacher-forced slopes are human-like.** Then the conditional
  is right and the loss happens in the closed loop, where the per-agent GRU is
  the only route from the current round to the next decision: everything the
  player just experienced must survive the update gate of a state that, in
  self-play, already encodes the model's own intent for the round. The change is
  then an **immediate-stimulus skip**: `op2` reads the post-`op1` embedding
  alongside the post-RNN one, so what I just gave and what I just got for it
  reach the next decision directly and not only through what I remember. It is
  the same shape of change as the parent's -- `op2` widens, the recurrent path
  is untouched -- and it is not scheduled sampling (§5's vetoed family): no
  own-rollout unrolling, no change to training time.

Either way the copula is then **recalibrated on the retrained trunk**, per PR
#166's lesson that separately calibrated latents do not compose and PR #179's
measurement that recalibration moved rho by -37%.

**Watch items (reported whatever the verdict, never claimed):** **RCC**, the
ceiling twin of RCB and the highest remaining row; **RCA** and **RSA**, the
other response rows the same channel feeds; the marginal C block **CA / CB / CD
/ CE / CF** and **CG**, which a retrained trunk re-randomises and which PR #179
bought at the noise ceiling; **SC**, the parent chain's success row; **PD**,
sensitive to the contribution slot's dependence structure through the
recalibration.

## 2. Plan

**Step 0 fired branch (b).** Teacher-forced on the human data, the parent trunk's
RCB statistic is **0.0930** -- *inside* the human-vs-human noise ceiling
(0.3479), a score of **0.267**, band <= 1 -- against **0.7973** for the same
trunk in the closed loop. Its within-band slopes are +0.1157 / +0.0791 /
+0.0425 / -0.1659 against human +0.1397 / +0.1038 / -0.0767 / -0.1615: neither
clause of (a) is met (0-4 is +0.1157, above the +0.10 threshold; fewer than two
bands sit below half the human value). SLURM 30266433, exit 0:0, 12 s; the
script's self-check reproduces all four human slopes and all four human bin
means to seven decimals on the identical population (weighted discrepancy
1.49e-07), so the comparison is like for like.

**What that means, and why the skip is the change.** The model is one
deterministic map from (this round's features `x_t`, the carried state
`h_{t-1}`) to next round's distribution. The same map produces a human-like
punishment response when `x_t` and `h_{t-1}` come from the human trajectories
and a quarter of it when they come from the model's own. The declaration's
cell-level decomposition already showed the `(contribution, punishment)`
composition is not the difference -- at the *same* cells the sim's responses are
wrong (0.5707 against the teacher-forced 0.0930). What is left is `h`: in
self-play the carried state leaves the manifold the model trained on, and the
current round's stimulus is swamped, because the per-agent GRU is the **only**
route from `x_t` to the readout. The skip opens a second route that does not
pass through the update gate of a drifted state.

**Behavioural rationale (one sentence, per §5):** what I just gave and what I
just got for it should reach my next decision directly, not only through what I
remember -- the row that should move is **RCB**.

| # | step | implementer |
|---|---|---|
| 1 | **The immediate-stimulus skip in `GraphNetwork`.** `src/aimanager/generic/graph.py`, constructor and `forward`. Add a `stimulus_skip: bool = False` flag. When on, `op2`'s `NodeModel` takes `x_features + hidden_size` (plus the vnode width, unchanged), and `forward` keeps the post-`op1` node embedding and concatenates it to the post-RNN embedding at the `op2` readout. Plain concatenation, no gate (§5: ties go to the simpler model). Nothing new is constructed, so with the flag off the build is bit-identical and today's artifacts load unchanged. The recurrent path, the edge model, the vnode and the joint head are untouched. | **Opus** |
| 2 | **Tests for the skip.** New `tests/skip/test_stimulus_skip.py`, modelled on `tests/vnode/test_group_vnode_graph.py`. Gates: (a) flag off -> `state_dict` shapes and forward outputs bit-identical to the flag absent; (b) flag on -> `op2`'s input width grows by exactly `hidden_size` and the skip carries gradient; (c) 24 single-round calls with `reset_rnn=False` reproduce one 24-round call, the simulation's calling convention; (d) a saved-and-reloaded model round-trips the flag. Run on Raven (PyG). | **Opus** |
| 3 | **The training config.** New `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_vnode_stimulus_skip.yml`, a copy of the parent's `..._group_vnode.yml` with `model_args.stimulus_skip: True` and `output_dir` -> `..._50ep_vnode_stimulus_skip`. Every other field byte-identical: seed 38381, 575 epochs, batch 4, lr 3e-4, 5-fold CV, same `x_encoding`, `shuffle_features`, data file and mask. | **Sonnet** |
| 4 | **Train the candidate trunk.** `AI_REMOTE_DIR='~/autoresearch/contribution-punish-onehot' scripts/train_cluster.sh ah <config>`. Record held-out log loss against the parent's 2.008563 and wall time against §5's ~33 min ceiling. | **Sonnet** |
| 5 | **Recalibrate and stamp the copula on the new trunk.** Per PR #166's lesson and PR #179's measurement that recalibration moved rho by -37%. New slurm pair copied from `scripts/artificial_humans/{calibrate,stamp}_copula_group_vnode.slurm` with the new paths; `contribution_copula_rho.py --roundtrip --preflight --write-params`, then `make_contribution_copula_artifact.py`. Report the new rho and phi against the parent's 0.0435568043640977 / 1.0 and whether the parent's value sits inside the new CI. | **Sonnet** |
| 6 | **Pre-simulation mechanism gate: the response at sim-visited states.** Extend `scripts/data_analysis/rcb_teacher_forced.py` with a mode that teacher-forces a trunk on a *simulation's* `per_round.parquet` trajectories instead of the human CSV. Two checks: (i) the **parent** trunk on the **parent's** sim trajectories must reproduce the parent's flat closed-loop slopes -- that validates the whole diagnosis, since a teacher-forced pass over the realised states is the closed loop's conditional expectation; (ii) the **candidate** trunk on those same states predicts whether the skip actually restores the response off the human manifold. Report both, and the candidate's human-data teacher-forced numbers too (they must not regress). | **Opus** |
| 7 | **Simulation.** New `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch.yml`, a copy of the parent's sim config with only the contribution artifact path swapped. Protocol untouched: 2 groups x 8 agents, 24 rounds, 100 episodes, seed 42, `save_per_round: true`, same switch predictor and same severity-copula punisher, single pairing. Run on Raven, fetch. | **Sonnet** |
| 8 | **Evaluate and record.** `python -m aimanager evaluate <sim config>` locally; fill the results table and the watch items; verdict straight from §2 against RCB 2.3151705700149083 and the mean ceiling 1.2087123341579042. | **Sonnet** |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-16 | (baseline) parent stack, PR #179 | RCB 2.3151705700149083 | 12/21 | 1.09882939472699 | baseline |
| 2026-09-16 | immediate-stimulus skip + copula recalibrated (rho 0.03949863621805423, phi 1.0) | **RCB 2.0867005460163575** | 12/21 | 1.1045228654552426 | **FAIL** |

**Gate 1 FAILS.** RCB 2.3151705700149083 -> **2.0867005460163575** stays in band
2-5: the resampled discrepancy falls 0.8053441343922023 -> **0.7258696472443746**
against a band edge of 0.6957104109931878, so the row misses by **4.34%** of the
statistic. The required reduction was 13.6%; the delivered reduction is **9.87%**,
about 73% of it.

**Gate 2 PASSES.** The 21-row mean rises 1.09882939472699 -> **1.1045228654552426**,
+0.52%, well inside the 1.208712334199689 ceiling. Rows <= 1 hold at 12/21.

Since gate 1 fails, the experiment is a `[FAIL]` whatever gate 2 does.

**The mechanism installed and is correctly signed in every band.** Slope of dc on
punishment received, within contribution band:

| band | human | parent | **candidate** | step 6's prediction | gap closed |
|---|---|---|---|---|---|
| 0-4 | +0.1397 | +0.0619 | **+0.0730** | +0.0715 | 14% |
| 5-9 | +0.1038 | +0.0115 | **+0.0541** | +0.0439 | 46% |
| 10-14 | -0.0767 | -0.0082 | **-0.0467** | -0.0223 | 56% |
| 15-19 | -0.1615 | -0.0366 | **-0.0253** | -0.0510 | **-8%** |

Three of the four bands improve, the 10-14 band's sign flips to human for the
first time in this campaign, and the 15-19 band moves the wrong way.

## 4. Notes

1. RCB is the only row >= 2 on this parent, so the target list is forced; there
   is no target-shopping decision to make.
2. The ratio the metric names is not the behavioural variable. Controlling for
   contribution and punishment, the human rate coefficient is **-0.6871** while
   punishment carries **+0.1337**: the rate enters with the wrong sign, and the
   apparent monotone rise across the rate bins is carried by punishment size
   together with the contribution level that the ratio's denominator mixes in.
   A rate feature is therefore ruled out on evidence as well as on §5.
3. The sim's sign is not wrong. The top rate bin's -0.151 is composition: that
   bin is 43% high-contribution rows, where the human mean is also negative
   (-4.56 at 15-19 with punishment 11+). Within contribution band the sim's
   signs match the human's in all four bands; only the magnitudes fail, at
   11-44% of human.
4. A cell-level decomposition shows the conditional response, not the cell
   composition, is the binding half: giving the sim's own composition the human
   conditional takes the scored statistic to 0.3731, a long way inside the
   0.6957 band edge, while giving the human composition the sim's conditional
   leaves 0.5707. Only about half the attenuation has to be closed.
5. The one-hot argument was tested before it was adopted rather than after. A
   width-20, single-tanh surrogate of the per-agent path learns the human
   within-band slopes from **numeric** inputs, with and without the trunk's real
   competitors present, so "a numeric-by-numeric linear input cannot form the
   response" is too strong as stated: the encoding improves the RCB bin gap
   (0.2226 -> 0.1433 with both inputs one-hot) but is not shown to be the
   binding constraint. Step 0 measures the trunk itself before the change is
   fixed; the decision rule is recorded above, before its numbers exist.
6. The sampler is not the cause, so there is no "fix the copula" direction here.
   Re-measuring the within-band slopes on PR #179's two diagnostic runs, which
   differ from the candidate only in the copula fields:

   | run | 0-4 | 5-9 | 10-14 | 15-19 | scored d |
   |---|---|---|---|---|---|
   | human | +0.1397 | +0.1038 | -0.0767 | -0.1615 | 0 |
   | candidate (rho 0.0436, phi 1) | +0.0619 | +0.0115 | -0.0082 | -0.0366 | 0.7973 |
   | no copula (weight-identical trunk) | +0.0542 | +0.0044 | -0.0278 | **+0.0258** | 0.7451 |
   | phi_hat 0.618 | +0.0393 | -0.0201 | -0.0209 | -0.0145 | 0.7029 |

   Stripping the latent entirely leaves the response just as flat -- and flips
   the top band's sign the wrong way. The attenuation is a property of the
   trunk in the loop, not of the shared latent sitting on top of it. (The
   phi_hat run's 0.7029 is closer to the 0.6957 edge on *worse* slopes, i.e.
   bin-composition luck, and PR #179 already ruled phi_hat out as a CG trade.)
7. Training cost is not a constraint here. Recent `train-ah` jobs on Raven
   (`architecture_node+edge+rnn__d+`, SLURM 30256024/30256025/30256343) run
   8m43s-11m12s, so §5's 3x budget is ~33 min; neither candidate change --
   a wider input encoding or a wider `op2` readout -- moves training time
   materially.
8. **Step 0 settles it: the conditional is not broken, the states are.** The
   trunk teacher-forced on human data scores RCB at **0.267** of the noise
   ceiling; the same weights in self-play score **2.3152**. Since a
   teacher-forced pass over a trajectory *is* that trajectory's conditional
   expectation, the entire 8.7x is carried by which `(x_t, h_{t-1})` the loop
   visits -- and note 4 already ruled out the `(contribution, punishment)`
   composition as the explanation. What is left is the carried state.
9. **The one-hot hypothesis is therefore answered in the negative, on the
   trunk itself rather than on a surrogate.** A representation change to
   `prev_punishment` addresses how well the conditional response is *learned*;
   step 0 shows the conditional response is learned to within the noise
   ceiling. There is nothing there to buy. This is the second independent
   measurement pointing the same way (note 5 was the surrogate) and it is the
   decisive one, because it is the actual trunk on the actual data.
10. **One real weakness in the branch-(b) story, recorded before the sim.** The
    10-14 band's sign is wrong even teacher-forced: the trunk gives **+0.0425**
    where humans give **-0.0767**, so the midpoint sign flip is genuinely absent
    from the conditional in that band (it is present at 103% of human in 15-19).
    A skip connection does not obviously repair a wrong-signed conditional. The
    bin-mean statistic hides this because the 10-14 rows are diluted across rate
    bins; it is the most likely reason the candidate could move less than the
    teacher-forced ceiling suggests, and it is the seed for a successor if this
    experiment lands short.
11. The step-0 script's alignment is pinned by four checks, not by argument: the
    structural lag identity `prev_contribution[:,:,1:] == contribution[:,:,:-1]`,
    the absence of any current-round feature in `x_encoding`, a lag profile
    `corr(E[c_t], c_{t+k})` peaking at the conditioning lag (-2: 0.8694,
    **-1: 0.9372**, 0: 0.8513, +1: 0.8129, +2: 0.7858), and the OLS slope of
    `E[c_{t+1}]` on `c_t` at **+0.7709** against the human **+0.7686**. The
    deliberately misaligned pairing is reported alongside and is visibly worse
    (weighted discrepancy 0.2487 vs 0.0930).
12. The 10-episode holdout is not usable for this measurement and is reported
    for completeness only: its own *human* slopes are +0.1958 / +0.3523 /
    +0.0087 / +0.1038, with no sign flip at all and n=35 in the top band. The
    union of the two splits -- the single copy, 50 episodes, n=2,660 -- is the
    canonical evaluation frame's population and is the number used above.
13. **The verdict, plainly: the diagnosis was right and the remedy was
    two-thirds of one.** Step 0 said the conditional is healthy and the loop
    loses it; step 6 confirmed that at the sim's own states; the skip then
    moved the loop's response in the correct direction in every band and still
    missed the band edge by 4.34%. Nothing here was a wrong turn that better
    execution would have avoided -- the mechanism is real and it is simply not
    large enough on its own.
14. **The pre-simulation gate predicted the outcome, and that is the most
    transferable result here.** Step 6 teacher-forced the candidate over the
    *parent's* realised states and predicted a statistic of 0.7098; the
    candidate's own closed loop delivered a raw 0.7182 (scored 0.7259). A
    forecast within 1.2% of raw discrepancy, made before any simulation was
    spent, from a 63-second job. Any contribution-slot experiment can now screen
    a trunk against RCB-family rows this way before committing a sim, and the
    same construction generalises to any row that is a conditional of the
    contribution model. That capability is worth more than this experiment's
    own number.
15. **Where the prediction was wrong is where the successor lives.** The two
    bands step 6 got right are 0-4 and 5-9 (predicted +0.0715 / +0.0439,
    realised +0.0730 / +0.0541). The two it got wrong diverged in *opposite*
    directions: 10-14 overshot toward human (-0.0223 predicted, -0.0467
    realised) while **15-19 undershot and moved away** (-0.0510 predicted,
    -0.0253 realised, human -0.1615 -- the only band that got worse than the
    parent's -0.0366). The high-contribution withdrawal response is the band
    the candidate's own rollout reaches differently from the parent's, and it is
    the one carrying 1.04 of raw human signal in the top rate bin.
16. **The collateral says what the skip actually traded.** Two band upgrades --
    CE 1.1107 -> 0.9098 and CF 1.0762 -> 0.8866, both 1-2 -> <= 1 -- and the
    whole marginal C block improves (CA -0.113, CB -0.123, CC -0.098,
    CD -0.126). Against that, three band downgrades: **CG 0.8990 -> 1.3101**
    (<= 1 -> 1-2, the parent's headline row), **RCD 1.3405 -> 2.2051**
    (1-2 -> 2-5, the largest single move in the table), and SC 0.9775 -> 1.0228
    (<= 1 -> 1-2, by 0.023). The pattern is coherent and it is the point: CG and
    RCD are the two rows that *depend on persistence* -- group-level spread
    accumulating over an episode, and a switcher carrying the receiving group's
    state. Handing the readout a route that bypasses the recurrent state buys
    immediacy and sells exactly that persistence. The individual-fit rows, which
    want immediacy, all improve.
17. **That trade is the successor: gate the skip instead of concatenating it.**
    §5's "ties go to the simpler model" chose a plain concatenation, and the
    result is a model that weights the immediate stimulus and the carried state
    at one fixed ratio for every round. The behaviour wants the opposite: the
    stimulus should dominate on the rounds where something happened to you --
    you were punished, your group changed -- and the memory should dominate
    otherwise. A scalar gate on the skip's contribution, learned from the same
    features, would let CG and RCD keep their persistence on quiet rounds while
    RCB gets its response on loud ones. The tie-break was the right call under
    the rule and the evidence now argues against it, which is the cleanest kind
    of finding to hand on.
18. **RCB is still the only row >= 2 in the candidate's table, and it now has
    company**: RCD 2.2051 joins it. A successor stacked on PR #179 should expect
    to defend both, and note that RCB alone at 2.0867 is now only 4.34% of
    statistic from its edge -- the closest this row has been.
19. The copula recalibration was correct to run and nearly a no-op: rho moved
    -9.3% and the parent's value sat *inside* the new CI, unlike PR #179's
    -37.4% with the old value above the upper bound. Worth recording as a
    counter-example to the reflex that a retrained trunk always needs a very
    different dose -- the rule is recalibrate and *look*, not recalibrate and
    assume.
