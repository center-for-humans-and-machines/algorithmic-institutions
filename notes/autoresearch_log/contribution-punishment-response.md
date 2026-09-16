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

_To be written after step 0._

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-16 | (baseline) parent stack, PR #179 | RCB 2.3151705700149083 | 12/21 | 1.0988293946890038 | baseline |

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
