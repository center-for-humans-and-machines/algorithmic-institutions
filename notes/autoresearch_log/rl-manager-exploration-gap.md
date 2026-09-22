# The manager that trains is not the manager we evaluate: exploration injects 1.43 punishment points per member-round

## Declaration

**Not a slot experiment, and nothing was run.** No artificial-human model, feature, config or line of training code changes. This branch re-reads metrics that six finished RL-manager runs already wrote and is not judged by the §2 gates of `notes/autoresearch.md`. No GPU time was spent; the whole result is fifty lines of pandas over committed parquets.

**Base.** `auto/rl-manager-two-worlds` at `f814c31` ("Three runs done; validity-conditioned probe and convergence read"), which carries the pool arm's three metrics parquets and the launch-guard evidence this note leans on for the timeout rate.

**What this branch adds to the data.** The per-capita arm's three metrics parquets, fetched from `~/repros/ai-runs/rl-percapita` and committed here for the first time. Jobs **30403651 / 30403686 / 30403808**, all `COMPLETED`, elapsed **06:16:38 / 06:28:36 / 06:25:00** — the three runs that `auto/rl-manager-percapita-reward` (PR #210) launched and handed back as `RUNNING`. Only the metrics are committed; the three 6.7 MB checkpoints stay on the cluster, because nothing here reads a policy's weights. The per-capita *configs* are deliberately not carried over: they belong to PR #210, and this branch is a reader, not a second copy of that arm.

**The question.** The manager explores with epsilon-greedy at a fixed 0.1 over 31 ordinal punishment levels, 0 to 30. A uniform draw over that range has expectation 15, so exploration alone puts 0.1 × 15 = 1.5 punishment points per member per round into the game. Every run's learned signal is smaller than that. If the arithmetic holds, the policy that generated every transition in the replay buffer and every state the artificial humans reacted to is a **materially more punitive manager** than the one the run reports — and the run reports the quiet one.

**Artifacts.** Script `scripts/data_analysis/rl_manager_exploration_gap.py`; tables and figure `plots/data_analysis/evaluation/rl_manager_exploration_gap/`.

## Measured

Everything in this section was computed from committed files. The window is `update_step >= 2980` — the 51 logged evaluation points from 2980 to 3980, the last 1000 update steps of 4000. Section 6 gives the sensitivity to that edge.

### 1. The knobs, read from the code and the configs

| what | where | value |
|---|---|---|
| epsilon | `manager.py:60`, `manager.py:83` | **0.1, fixed** |
| annealing | nowhere | **none** — `self.eps` is assigned once in `__init__` and read once in `get_action`; `grep -ri "anneal\|decay\|eps_start\|eps_end\|schedule" src/aimanager/` returns only the rule-based manager's unrelated `decay` rule and one comment in `generic/graph.py` |
| exploratory draw | `manager.py:79` | `th.randint(0, n_actions, size=greedy_action.shape)` — **uniform over all 31 levels**, drawn independently of `greedy_action`, not local to it |
| levels | all six configs | `n_punishments: 31`, so 0…30, mean draw **15** |
| steps | all six configs | `n_update_steps: 4000`, `eval_period: 20` |

So the exploration is uniform over the full ordinal range for all 4000 steps. A manager that has learned to punish 1 and a manager that has learned to punish 25 explore identically, and at step 3999 exactly as much as at step 0.

**Both rollouts are in the same parquet, and only one of them trains the network.** `rl_manager.train_manager` calls `run_batch` twice: once every step with `on_policy=False` and the replay buffer attached (logged when `update_step % 20 == 0`), and once every 20 steps with `on_policy=True` and `replay_mem=None`. The `sampling` column separates them. **The replay buffer therefore contains eps-greedy transitions and nothing else**, and `greedy` rows are a read-only measurement of a policy that never generated a single training target.

### 2. The two policies, over the last 1000 update steps

| arm | seed | evaluated (greedy) | behaviour (eps-greedy) | ratio | gap |
|---|---|---|---|---|---|
| pool | 42 | **1.8060** | **2.9952** | 1.66 | 1.1892 |
| pool | 43 | **1.0760** | **2.2551** | 2.10 | 1.1791 |
| pool | 44 | **1.7895** | **3.0122** | 1.68 | 1.2227 |
| per capita | 42 | **1.6021** | **2.8276** | 1.76 | 1.2255 |
| per capita | 43 | **0.2264** | **1.4951** | 6.60 | 1.2687 |
| per capita | 44 | **1.2713** | **2.4636** | 1.94 | 1.1924 |

The gap sits in **1.179 – 1.269** in every one of the six runs, against learned signals spanning 0.23 to 1.81. Exploration is **0.79× to 6.33×** the size of the whole policy it is added to.

For scale: the human managers punish **1.79** per agent-round (parent log, measurement plan item 1, 18,386 rows with a valid manager input), and the artificial punisher holding the other group in these same runs punishes **1.503 – 1.631**. Five of the six behaviour policies punish more than that human mean; of the six evaluated policies only one does, and only by 0.016.

**Per-capita seed 43 is the sharpest case, and the coincidence is exact.** Its evaluated policy punishes **0.2264**. Its behaviour policy — what the artificial humans in its group actually experienced for all 4000 update steps — punishes **1.4951**, which is within **0.008** of the clone's own level in that same run (1.5029 in the greedy rollout). The policy that run reports was never the manager its own training data describes.

### 3. The mixture arithmetic, and what the free-punishment fix does to it

`environment.punish` zeroes a punishment aimed at a player who gave no input (`auto/free-punishment-fix`, PR #208), and it zeroes the exploratory draws exactly as it zeroes the greedy ones. So the realised behaviour mean is

```
E[valid × ((1-eps)·a_greedy + eps·U)]  =  (1-eps)·g + eps·v·15
```

and the predicted gap is `1.5·v − eps·g`, not a constant. With `v` from the launch-guard evidence of this same world — **35,205 of 36,864 agent cells gave input, v = 0.9550** (`guards_after_fix.json`, `guard2`) — exploration delivers **1.4325**, not 1.5, per member-round.

Residuals of the measured gap against each prediction, **at round 0** — the one round where the identity is entitled to hold, for the reason section 4 gives:

| arm | seed | residual vs `1.5 − 0.1g` | residual vs `1.4325 − 0.1g` |
|---|---|---|---|
| pool | 42 | −0.0815 | **−0.0140** |
| pool | 43 | −0.0494 | **+0.0181** |
| pool | 44 | −0.0865 | **−0.0190** |
| per capita | 42 | −0.0616 | **+0.0059** |
| per capita | 43 | −0.0707 | **−0.0031** |
| per capita | 44 | −0.1019 | **−0.0344** |

Mean residual **−0.0753** without the zeroing, **−0.0078** with it, largest |residual| **0.0344**. The zeroing is not merely a plausible explanation of the round-0 residual; it is very nearly all of it, and the sign flips from uniformly negative to scattered about zero once it is in the prediction.

### 4. Round 0 is where the identity has to hold, and it does

`env.reset()` calls `update_contribution()` — round-0 contributions and validity are drawn **before the manager acts**. So at round 0 the greedy rollout and the eps-greedy rollout face the same state distribution and the mixture identity is a clean test. From round 1 on the two rollouts are on different trajectories and it need not hold at all.

Fitting `gap = a·g + b` across the six runs, which assumes neither epsilon nor the timeout rate:

| window | slope | theory | intercept | implied `v` | guard `v` | R² |
|---|---|---|---|---|---|---|
| **round 0** | **−0.1057** | −0.1000 | **1.4340** | **0.9560** | 0.9550 | **0.978** |
| episode mean | −0.0308 | −0.1000 | 1.2528 | 0.8352 | 0.9550 | 0.309 |

At round 0 the punishment metrics alone **recover epsilon to 0.006 and the timeout rate to 0.1 percentage points**, from six runs whose greedy levels span 0.14 to 3.00. That is the free-punishment mechanism measured rather than asserted.

Over the episode the identity breaks: the slope collapses by a factor of three and R² falls to 0.31. Against the zeroing-corrected prediction the episode residuals are **−0.031 to −0.146, mean −0.090** — an order of magnitude larger than the round-0 residuals, appearing at round 1 and growing. The zeroing cannot explain that, because the zeroing is already in the prediction. What is left is the two rollouts standing on different state distributions, which is the finding itself, measured as a number.

### 5. Contribution: the lift is caused by the exploration, and round 0 proves it

Contribution is higher under the behaviour policy in all six runs.

| arm | seed | greedy | behaviour | episode gap | **round-0 gap** |
|---|---|---|---|---|---|
| pool | 42 | 7.4743 | 7.6935 | +0.2192 | −0.0100 |
| pool | 43 | 8.2377 | 8.5537 | +0.3160 | +0.0296 |
| pool | 44 | 8.4378 | 8.6123 | +0.1745 | +0.0304 |
| per capita | 42 | 7.5080 | 7.7872 | +0.2793 | +0.0052 |
| per capita | 43 | 7.9934 | 8.2964 | +0.3030 | +0.0554 |
| per capita | 44 | 8.2752 | 8.3827 | +0.1075 | +0.0036 |

The round-0 column is the control, and it is the one that makes this causal rather than correlational: at round 0 contributions are drawn before any manager has acted, so the gap **must** be zero, and it is (−0.010 to +0.055). The gap then opens monotonically within each four-round block and reaches **+0.22 to +0.49** by round 23, resetting partially at each `switch_every: 4` reshuffle (`gap_by_round.jpg`, right panel). Exploration raises contribution, downstream, within the episode.

So the replay buffer — which holds eps-greedy transitions and nothing else — systematically carries a punishment-raises-contribution relationship at a punishment level the evaluated policy does not produce.

### 6. The seed spread under each policy, and the window edge

| statistic | evaluated | behaviour |
|---|---|---|
| sd across the six runs | **0.5984** | **0.5806** |
| range | 1.5796 | 1.5171 |
| coefficient of variation | 0.4620 | 0.2315 |

The **absolute** spread is the same under both policies; only the relative spread halves, and it halves because exploration adds a near-constant 1.2 to every run. Within arms, pool sd 0.417 and per-capita sd 0.718.

The spread does not close with training. Greedy mean punishment per 1000-step quarter, sd across the six runs: **1.119 / 0.486 / 0.559 / 0.603**. It narrows once as the runs leave their initialisations and then widens again over the last 2000 steps — consistent with the parent log's note 27, which found `q_mean` still climbing inside the final quarter.

**Window sensitivity.** The table in section 2 is at `update_step >= 2980`. At `>= 3000` (50 points instead of 51, the convention the parent log's note 27 used) no mean moves by more than **0.0128** and no ratio by more than **0.0067**; three of the six rows change in the second decimal. `window_sensitivity.csv` carries both.

## Inferred, not measured

Flagged separately because none of it has been observed.

1. **This is a candidate mechanism for the seed spread, not a demonstration of one.** What is measured is that each run's evaluated policy was never the policy that filled its replay buffer, chose its TD targets or set the state distribution its value function was fitted on. That makes "seed 43 found a quieter optimum" and "seed 43's optimum is poorly estimated because it was never visited" both live readings of the same number, where before only the first was on the table. **It does not adjudicate between them.** Section 6 is a mild caution in the other direction: the absolute spread is essentially identical under the two policies, so exploration is not manufacturing or masking the *size* of the spread. The experiment that would adjudicate is in the successor list, and it is one config key.

2. **"Levels the greedy policy never produces" is not measured and the metrics cannot measure it.** `rec_keys` logs round means only, so the action *distribution* over the 31 levels is absent from every parquet in this branch. What is measured is that exploration places 9.55% of realised agent-rounds on a uniform draw, of which 3.39 percentage points land at level 20 or above. Whether the greedy policy ever emits a high level is unknown, and a Markov bound from the means is far too loose to settle it (at level 20 it permits up to 9.0% for pool s42 against exploration's 3.4%). The honest statement is that the buffer contains high-punishment transitions **by construction**; that the greedy policy contains none is a guess.

3. **Nothing here says the gap is harmful.** Off-policy Q-learning is built to evaluate a policy other than the one acting; that is what the target network and the argmax are for. The claim is about magnitude — exploration the size of the entire learned signal, undecayed, over the full action range — not about the existence of a behaviour/target split. Whether a smaller or annealed epsilon produces a better manager, a worse one, or the same one is untested.

4. **The two arms are pooled here, and that is a reporting choice with a cost.** Six runs are used as six points because the mechanism is a property of the exploration schedule, which is byte-identical across arms. Every quantity is also reported per run, and no arm-vs-arm claim is made anywhere in this note; that comparison belongs to PR #210's successor and needs the per-capita arm's own behavioural table first.

## Status: measured and closed, no change proposed

This note changes no code and proposes none. It records a property of six runs that are already finished, and it commits the three per-capita metrics parquets so that the property can be re-derived without a cluster read.

## Successor

1. **The cheapest decisive experiment is one config key: re-run one seed at `eps: 0.01`**, or with a linear anneal to it, and read the same table. If the greedy policy lands where this run's greedy policy did, the exploration gap is a reporting curiosity. If it lands where this run's *behaviour* policy did, every policy in PRs #204 and #210 has been read off the wrong rollout. At ~6 A100-hours a seed this is the cheapest open question in the manager programme.
2. **Log the action histogram**, not just the round mean. Adding a per-level count to `rec_keys` costs one tensor `bincount` per round and settles inferred item 2 permanently, for every future run.
3. **Re-read the convergence question with this in mind.** The parent log's note 27 reports `q_mean` still climbing at step 4000 and reads it as under-training. A value function fitted on a state distribution 1.2 punishment points away from its own policy's is a second explanation for a `q_mean` that does not settle, and the two are separable by the `eps: 0.01` run above.
4. **The cross-evaluation is unaffected and should be said to be.** `simulate.py` runs a saved manager greedily, so `plots/simulation/24_rl_new_clones_cross_eval/` reports the greedy policy — the correct one for a behavioural claim. What this note puts in question is the *training*, not the evaluation of what training produced.

## Notes

1. **The per-capita parquets were fetched, not regenerated.** `rsync` from `~/repros/ai-runs/rl-percapita/artifacts/manager/rl_new_clones_percapita_s4{2,3,4}/metrics/`, the isolated dir PR #210 names in its note 5. That dir is scheduled for deletion when #210 closes, which is the reason the files are committed here rather than left on the cluster.

2. **Why the guard's timeout rate and not a fresh one.** `v` is needed to turn "1.5" into "1.4325", and re-measuring it would mean instantiating the validity model — PyG, and therefore the cluster. `guards_after_fix.json` is the same world, the same validity model and the same batch shape, was produced to certify these very runs, and is committed. Section 4 then makes the borrowing safe rather than load-bearing: the round-0 fit **re-derives** `v = 0.9560` from the punishment metrics alone and agrees with the guard's 0.9550 to 0.1 percentage points, so the guard number is confirmed by the data it is being applied to.

3. **The brief this note was written from predicted a gap of "about 1.35" and the measured gaps are 1.18–1.27.** Both are right about different things: 1.35 is `1.5 − 0.1·g` evaluated at roughly the mean greedy level, but the prediction is not a constant — it runs from 1.32 at pool s42 to 1.48 at per-capita s43, and the *measured* gap barely moves at all. That near-constancy is itself informative and is what section 4 explains: the `−0.1·g` term is real (the round-0 slope is −0.106) but at the episode level it is almost cancelled by the trajectory divergence, whose size happens to scale with `g` in the opposite direction.

4. **`scripts/data_analysis/rl_manager_exploration_gap.py` re-checks its own constants.** `EPS` and `N_PUNISHMENTS` are asserted against whichever of the six configs are present on the branch (three, here) rather than trusted, so the script fails loudly if it is ever pointed at runs with a different exploration schedule. The per-capita configs are absent by design and the assertion simply skips them; PR #210's section 3 establishes that the two arms differ in exactly `reward_mode`, `job_id` and `output_dir`.

5. **The sawtooth in the contribution panel has a name.** `switch_every: 4`, so rounds 4, 8, 12, 16 and 20 reshuffle group membership. The contribution gap collapses at each reshuffle — briefly going negative around round 4 in every run — and re-opens over the following three rounds, with the envelope growing across the episode. Group composition is the channel that resets; the punishment differential is not.
