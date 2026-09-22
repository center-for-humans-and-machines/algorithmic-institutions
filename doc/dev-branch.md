# The `dev` integration branch

`dev` is the shared base: the work a future experiment should inherit without having to think about it. Defect fixes, shared tooling, metric and protocol definitions, and additive capability that is off by default. It is built off `main` (`617683c`) by merging 24 pull requests in 26 merges (#213 and #216 moved mid-integration and were merged twice — §5), which carry 29 pull requests in total once each stack's ancestors are counted.

The rule applied to every branch below: **a branch belongs if inheriting it silently is the right default; it does not belong if it changes model behaviour on the strength of an experiment that failed.** Where a recorded verdict and that rule disagree, the verdict is quoted and the disagreement is argued rather than hidden. Verdicts are taken from each branch's own `notes/autoresearch_log/*.md` body, not from its branch name or PR title.

**Final test result: 721 passed, 4 skipped, 0 failed** on Raven (`src/ tests/ scripts/tests/`), isolated remote dir `~/repros/ai-runs/devint-a9837`. The 4 skips are environmental: `tests/skip/` and `tests/vnode/` shell out to `git show` to diff against the pre-change `graph.py`, and the rsynced remote directory is not a git repository. Nothing is skipped for a code reason.

---

## 1. The disposition table

`in` means the branch's commits are ancestors of `dev`. `ancestor` means it was not merged by name but its content arrived underneath a branch that was — which is not a lesser status, and is called out because it is the part a reader would otherwise miss.

### In

| PR | branch | recorded verdict | why it is in |
|---|---|---|---|
| 160 | `auto/punisher-severity-copula-v2` | SUCCESS | Severity-copula sampling band-upgraded PD 2.93 -> 1.53; both gates pass. Ancestor of #184. |
| 165 | `auto/contribution-herding-copula-v2` | SUCCESS | Episode-persistent group copula band-upgraded CG (9.81 -> 4.16) and RCD. Ancestor of #184. |
| 171 | `auto/switch-joint-exodus` | SUCCESS | Joint leaver-count head, SC 2.19 -> 1.15. Ancestor of #184; also carries the shared cluster-script fixes. |
| 179 | `auto/contribution-group-vnode` | SUCCESS | Per-group virtual node, CG upgraded two bands, 22-row mean 1.30 -> 1.10. Ancestor of #184. |
| 181 | `auto/contribution-punishment-response` | **FAIL** | Gate missed (RCB 2.32 -> 2.09, stayed in band). In anyway: the code is one `stimulus_skip=False` keyword on `GraphNetwork`, and an artifact saved without the key loads with the skip absent. See §3.1. |
| 184 | `auto/punisher-current-contribution` | REBASELINE | The punisher now reads the round it is punishing. A correctness fix with three independent confirmations; every stack's rows move. |
| 185 | `reports/autoresearch-atlases` | no gate | Two self-contained HTML reports plus their generators. Changes no model. |
| 189 | `docs/post-rebaseline-program` | no gate | The gate protocol itself: the noise freeze, noise-aware symmetric gates, six-model frontier baseline. Definitions, which is exactly what a shared base is for. |
| 192 | `auto/punisher-ceiling-fix` | **FAIL** | Gate missed on a band boundary, but the mechanism is verified: P(punish \| c=20) 12.2% -> 4.0% against the human 3.9%. A correctness fix that happens not to cross a line. Ancestor of #196/#197. |
| 196 | `auto/punisher-timeout-feature` | **FAIL** | The declared borderline. Its own log: "the data-handling premise is right and the fix is correct ... the feature earns its place in the linear punisher only." See §3.2. |
| 197 | `auto/sim-timeout-imputation` | **FAIL** | Adds `served_state()`: the contribution and switch models are served the recorded 0. Variety of reached states 18.9 -> 21.4 against the human 27.9, with nothing retrained. The best result of the campaign. |
| 199 | `metrics/rce-response-slope` | tooling | The RCE row, which takes the suite from 21 rows to 22. Verified present: the suite registers 22 rows on `dev` and RCE is among them. |
| 200 | `results/september-measurements` | RESULT | Six measurements consolidated to 73 files. Two behaviour-neutral source additions; changes no model. Carries #183/#186/#187/#188/#191/#195. |
| 202 | `auto/group-drift-gain` | RESULT | Durable null: the late-divergence failure is not a gain problem (0.9098 human vs 0.8973 sim, 0.89 reseed sd), robust across seven specifications. **Deviation from the brief's prior — see §4.** |
| 203 | `auto/shared-signal-gain` | RESULT | Durable null: "the hypothesis is dead" — the contributor follows its group as strongly as people do. **Deviation from the brief's prior — see §4.** |
| 204 | `auto/rl-manager-two-worlds` | RESULT | Already an integration node for #205/#206/#207/#208, and the required base for #210/#211/#213/#215/#216. Its own additions are `linear_opponent.py` and tests. |
| 205 | `auto/rl-manager-timeout-view` | FIX | The RL manager reads `served_state()` too — the third consumer of the same defect. |
| 206 | `auto/manager-common-pool-reward` | FIX | Carries the payoff correction (a timed-out player is still paid) as well as the `common_pool` reward mode. |
| 207 | `auto/rule-based-manager-sweep` | RESULT | Introduces the reusable `RuleBasedManager` family, a config generator and a sweep report — capability, not only numbers. |
| 208 | `auto/free-punishment-fix` | fix | A punishment aimed at a player who gave no input is charged as 0, as the game charged it. Explicitly requested. |
| 209 | `auto/rule-vs-clone-paired` | RESULT | The paired competition harness's configs and report. Reuses the existing `pairings` mode; adds no new simulation mode. |
| 210 | `auto/rl-manager-percapita-reward` | LAUNCHED | Adds `common_pool_per_capita` as a fourth value of `REWARD_MODES`. Opt-in; the env default stays `"sum"` and the running arm's configs are unchanged. |
| 211 | `auto/rl-manager-exploration-gap` | RESULT | Re-analysis of six finished runs. No source change at all. |
| 212 | `auto/rl-manager-annealed-local` | RESULT | Annealed, local epsilon-greedy. Off by default: with `eps_final` / `eps_anneal_steps` / `explore_sigma` unset the `Exploration` object is the constant-eps uniform resample that was there before. |
| 213 | `auto/rl-manager-bootstrapped-dqn` | LAUNCHED (runs since finished) | Bootstrapped ensemble DQN. Off by default: `n_heads` absent or 1 is `th.equal` to the pre-bootstrap agent, asserted by its own tests. |
| 214 | `auto/rl-manager-evolution-strategies` | RESULT | Evolution strategies, in its own `es_manager.py` and `manager/run_es.py`. Reaches the shared code only through `env.served_state()`. |
| 215 | `auto/contributor-punishment-targeting` | RESULT | Interventional probe: 21.9% of the human targeting gradient survives. No source change. |
| 216 | `auto/rl-manager-param-noise` | LAUNCHED | Parameter-space noise. Off by default: `param_noise=None` and "a zero scale reproduces the existing agent exactly". |
| 217 | `auto/rule-inverted-targeting` | RESULT | Extends `RuleBasedManager` with the `inv_threshold` / `band` family plus an `--inverted` generator flag. Misaimed punishment costs 22-42 pool points. |

### Out

| PR | branch | recorded verdict | why it is out |
|---|---|---|---|
| 98 | `97-50ep-training-data-audit` | pre-campaign | A data audit whose central finding — `prev_punishment` is uninformative — is an artifact of the wrong-round alignment #184 fixed. Superseded by its own premise. |
| 100 | `99-rule-based-manager-strategy-testing` | pre-campaign | "Every rule strategy loses to baseline." #207 measured the opposite on the corrected stack. Superseded, and its four source edits come off a pre-rebaseline base. |
| 144 | `auto/contribution-self-history-dropout` | FAIL | Self-history dropout worsened RCA monotonically (2.035 -> 4.142). Closed "FAILED as declared." |
| 145 | `auto/switch-contrib-features` | SUCCESS | A real success, but on the pre-v2 switch lineage that #171's joint exodus head replaced. Superseded, not refuted. |
| 147 | `auto/contribution-cg-selfdrop` | SUCCESS | Same: a success on the pre-v2 contribution lineage superseded by #165 and #179. |
| 148 | `auto/contribution-prev-onehot` | SUCCESS | Prev-contribution one-hot, band-upgraded RCA. Superseded lineage; nothing downstream builds on it. |
| 149 | `auto/contribution-cg-copula` | FAIL | Stopped at preflight: the independence-floor spread ratio was already 0.784 against the closed loop's 0.59. |
| 150 | `auto/switch-herding-copula` | SUCCESS | Its band position did not survive an isolated re-run — #162 reproduced it exactly and showed the crossing is RNG-context-dependent. |
| 151 | `auto/contribution-gaussian-mlp` | FAIL | RCA band upgrade killed by CG exploding 3.98 -> 6.58; stack mean rose. |
| 152 | `auto/punisher-ar-gnn` | FAIL | Autoregressive punisher regressed PD 1.53 -> 1.71. No band upgrade reachable. |
| 153 | `auto/contribution-peer-attention` | FAIL | Only two of three declared targets; ruled FAIL without a Stage 2. |
| 154 | `auto/contribution-xgboost` | FAIL | Best likelihood on record, but lost the GNN's marginal C-block fit both times. |
| 155 | `auto/contribution-cont-target` | FAIL | Unscorable: the deterministic regression head empties the full-contributor stratum and RCC hard-fails. |
| 156 | `auto/contribution-mlp-regressor` | FAIL | RCD upgraded, RCA regressed past band 5; rows <= 1 fell 11 -> 8. |
| 157 | `auto/contribution-conformity-mixture` | FAIL | CG and RCD moved within band; arm B's RCD regressed past baseline. |
| 158 | `auto/contribution-type-latent` | FAIL | CG 9.85 -> 9.66 and rows <= 1 fell 11 -> 10. |
| 159 | `auto/contribution-group-latent` | FAIL | Best CG on record at the time (8.01) but no band upgrade; the joint finetune only overfit. |
| 161 | `auto/punisher-ar-gnn-v2` | FAIL | PD 2.61 -> 2.06, missed the band boundary by 0.057. |
| 162 | `auto/switch-herding-copula-v2` | FAIL | SC 2.82 -> 2.19, within band. Its value is the diagnostic, which #200's lineage already records. |
| 163 | `auto/contribution-cg-schedsamp-v2` | FAIL | Scheduled sampling produced no band upgrade; the p50 arm breached the gate-2 ceiling. |
| 164 | `auto/punisher-ar-copula` | SUCCESS | PD 2.06 -> 1.28, but on the AR-GNN punisher lineage (#161) that the frontier did not take; #160's severity copula is the one on `dev`. |
| 166 | `auto/switch-herding-copula-v3` | FAIL | The two latents do not compose: SC 2.19 -> 2.97 and the mean worsened. |
| 167 | `auto/contribution-gaussian-mlp-v2` | SUCCESS | Head of the parallel Gaussian-MLP family. #191 later showed that head extrapolates worse in closed loop, which "kills the combined head as motivated". |
| 168 | `auto/switch-herding-copula-recal` | FAIL | All four arms band-upgrade SC and all four raise the mean; gate 2 fails on every one. |
| 169 | `auto/switch-exodus-count` | FAIL | Stopped at the preflight gate: the oracle count distribution reached 5.76 against the declared 5.99. |
| 170 | `auto/contribution-gmlp-group-copula` | SUCCESS | On the Gaussian-MLP family (#167). Same reason as #167. |
| 172 | `auto/switch-joint-exodus-gmlp` | SUCCESS | #171's head reimplemented on the Gaussian family. `dev` carries #171 itself. |
| 173 | `auto/contribution-group-size` | FAIL | CE 1.105 -> 1.050, no band upgrade. |
| 174 | `auto/switch-exodus-k-onehot` | SUCCESS | Gaussian-family only; its k-one-hot did not carry to the frontier stack when ported (#190). |
| 175 | `auto/contribution-size-onehot` | FAIL | Both declared targets stayed in band. |
| 176 | `auto/contribution-arrival-tenure` | FAIL | RCD short of its boundary by 0.053. |
| 177 | `auto/contribution-inflated-gmlp` | SUCCESS | Gaussian-family only. Its salvageable piece, corner inflation, is recorded in #191 and so reaches `dev` through #200. |
| 178 | `auto/contribution-pna-aggregation` | FAIL | PNA aggregation drove CG the wrong way, 4.27 -> 5.22. |
| 180 | `auto/switch-round-onehot` | FAIL | All three of the ruling's criteria fail; SB left its band. |
| 183 | `rcb-holdout-teacher-forced` | RESULT | **Content is on `dev`**, condensed into #200. The branch itself is the evidence trail. |
| 186 | `auto/copula-closed-loop-variance` | RESULT | **Content is on `dev`** via #200. |
| 187 | `auto/copula-missing-state` | RESULT | **Content is on `dev`** via #200. |
| 188 | `auto/copula-seed-ensemble` | RESULT | **Content is on `dev`** via #200. Note its log contains hypothetical "would be a `[FAIL]`" phrasing that is not its verdict. |
| 190 | `auto/switch-kexo-port` | FAIL | SC improved within band, RCD moved the wrong way, and the protected RCE row's 10-14 slope eroded to 34% of baseline. |
| 191 | `auto/head-state-spread-diagnostic` | RESULT | **Content is on `dev`** via #200. |
| 193 | `auto/punisher-contribution-encoding` | FAIL | Both declared targets moved the wrong way and the protected RCE row eroded, despite a record 22-row mean. |
| 194 | `auto/contributor-ceiling-indicator` | FAIL | RCC 8.6% short of its boundary and the protected RCE row dropped a band. |
| 195 | `auto/seed-spread-noise-floor` | RESULT | **Content is on `dev`** via #200. |
| 198 | `auto/contribution-copula-recalibrated` | FAIL | A true null: refitting returned the identical rho, the model file is byte-identical and all 22 rows moved by exactly zero. Already condensed onto `main` as `c67308b`. |
| 201 | `auto/contributor-gated-skip` | FAIL | Refuted at the teacher-forced screen before any simulation: the gate collapses toward zero and RCE's middle bands move away from the human. |

---

## 2. Conflicts, and how each was resolved

Six merges conflicted. Every resolution is also written into the merge commit it belongs to.

**#208 against #206** — `src/aimanager/manager/environment.py`, one docstring hunk on `Environment.punish`, exactly as #208's log predicted. Both code changes merged cleanly and both are present: #208's `th.where(contribution_valid, punishment, 0)` and #206's `compute_reward_per_group` call. Resolved by keeping both docstring paragraphs; #206 documents where the reward is computed, #208 why the punishment is zeroed, and neither contradicts the other.

**#204 against the above** — the same docstring. #204 had already reconciled #206 and #208 itself, in the other order and with an added ordering note ("the charged ones, now that the zeroing above happens first"). Took #204's, which is the author's own reconciliation and states one more true thing. No code hunk conflicted.

**#216 against #213** — `manager.py` and `rl_manager.py`, three hunks each. Two additive exploration mechanisms written against the same base and never against each other, so every hunk was a union rather than a choice: both kwarg sets kept, `get_action` ordered param-noise then bootstrap then the unchanged eps-greedy dither, the `sampling` column extended to name all three arms, both rollout-tail blocks kept, and #213's order-preserving metric-melt loop extended with #216's prefix. With both mechanisms off the path is the one that was there before either arm.

That merge then broke six tests, three from each arm — see §3.3. They are fixed in a separate commit rather than buried in the merge.

**#212 against #213** — `manager.py` (four hunks) and `rl_manager.py` (one). One is a genuine name collision, not a text overlap: #212 names its `Exploration` **object** `self.exploration`, and #213 uses that same name for its mode **string** (`'eps_greedy'` / `'bootstrap'`), which is a config key under `manager_args` and is read by `rl_manager.run_batch` through `getattr`. Both cannot hold the name. Renamed #212's to `self.explorer`, because it is a pure runtime attribute built from `eps_final` / `eps_anneal_steps` / `explore_sigma` and has no config key of its own — the rename touches no YAML and no checkpoint, three call sites in `src/` and two in `scripts/rl_anneal_local/guard.py`. Renaming #213's would have changed a config key instead.

The other hunks: the `get_action` signature keeps both `head=` and `update_step=`; and the inline eps-greedy dither is deleted rather than kept beside `Exploration`, since #212 moved it wholesale into that class.

**#200 against #197** — add/add on `scripts/data_analysis/copula_closed_loop_variance.py`. #197 forked #186's script at the same path to read its own 2x2 of (serving fix off/on) x (copula off/on); #200 carries #186's original repointed at `main`. Both blobs were diffed in full: they differ only in the docstring, `OUT`, `ARMS`, `LABELS`, the palette and the suptitle. Every analysis function is byte-identical, including `teacher_force`, which `scripts/data_analysis/head_state_spread.py` imports "verbatim (PR #186)" — so that import behaves the same either way. Took #197's, because it is what every branch merged after #197 already carries; taking #200's would have silently reverted the file for the whole post-197 lineage. Regenerating #186's own figure is an edit of four constants, and #200's exact blob stays reachable through the merge's second parent.

**#185 against #200** — add/add on both atlas generators, the conflict #200's body predicted. Took #200's, which is what that body says to do: they are #185's generators with the 29 source paths repointed off six open branches onto one (named by `SEPT`, defaulting to `main`) plus a file-cache key changed from basename to full path. #200 verified both reports rebuild byte-identically against them, matching exactly the HTML #185 commits — so the reports taken from #185 are the output of the generators taken from #200. The three HTML files merged clean.

---

## 3. Judgement calls worth disagreeing with

### 3.1 The frontier lineage carries two failed gates, and that is unavoidable

This is the structural fact that shapes everything above. `auto/punisher-current-contribution` (#184) — a near-certain include, and the rebaseline everything downstream is measured against — has #181 in its ancestry, and #197 has #192 and #196 in its. You cannot take the defect fixes without taking those three failed experiments, short of rebuilding the lineage commit by commit.

That is acceptable here, but only because of what those three actually changed, which was checked rather than assumed:

- **#181** modifies exactly one file, `src/aimanager/generic/graph.py`, adding a `stimulus_skip=False` keyword. Everything else it adds is under new names — a new artifact directory, a new training config, a new sim directory. It modifies no existing config and no existing artifact. An artifact saved without the key loads with the skip absent.
- **#192** adds the `contribution_max` ceiling indicator, which is a feature made *available* in the grid and *used* by a new `_ceiling` config. Its gate failure was a band boundary, not a refutation; the mechanism moved P(punish \| c=20) from 12.2% to 4.0% against the human 3.9%.
- **#196** is §3.2.

So none of the three changes the behaviour of a model that does not opt in. If the maintainer disagrees with any one of them, the disagreement cannot be actioned by dropping that branch from `dev` — it needs the lineage rebuilt, and that is a much larger job than this one.

### 3.2 #196, the punisher timeout feature: merged whole, because the code already encodes the asymmetry

Its own verdict section is unambiguous, and worth quoting rather than paraphrasing:

> **[FAIL]** — gate 1 (RCC < 1.0) not cleared ... The change is nevertheless a **correct** data-handling fix with a real, independently predicted gain in the linear punisher, and should be kept on that family; it should not be carried into the graph punisher.

and, separately: "**Do not put `contribution_valid` in the graph punisher.**"

The branch contains two separable things, and only one of them is family-specific:

1. **Serving the recorded 0** on both punisher paths (`api_manager.create_data`, `linear_ah._pool_from_rounds`) and carrying the env's realised validity flag through `simulate.make_round`. This is the same correctness principle as #197, #205 and #208 — the value the game actually charged — and the accounting identity holds on all 4,512 group-rounds under 0 and fails on all 516 timeout group-rounds under the imputed 9. It is not family-specific and it belongs.
2. **The `contribution_valid` feature channel**, which is what the graph family must not have.

I merged it whole, and then checked that (2) is actually off for the graph family on `dev` rather than trusting it. It is: `contribution_valid` appears in exactly one punisher training config, `rnn_edge_50ep_doubled_timeout.yml`, which is the opt-in experimental arm. The frontier GNN punisher config `rnn_edge_50ep_doubled_ceiling.yml` carries `contribution` and `contribution_max` and **not** `contribution_valid`. On the linear side, `scripts/baselines/punishment_baseline.py` has it in `FEATS`, which is the family the log says should keep it, and `handcrafted_grid.py` makes it legal for the punishment target and illegal for the contribution target.

So the asymmetry the log asks for is expressed where this project expresses model choices — in the configs — and merging the code does not turn it on for the graph punisher. The alternative, surgically reverting the `api_manager` hunk, would have diverged `dev`'s code from every artifact and evaluation measured on it, and would have deleted a correctness fix to avoid a feature that is already off.

**What would change this call:** if the maintainer wants the graph punisher never to be *able* to read `contribution_valid`, that is a one-line deletion in `handcrafted_grid.py`'s legality set plus removing the `_timeout` config — not a change to `dev`'s merge structure.

### 3.3 The exploration arms broke each other, and the fix is mine

The four arms (#212, #213, #214, #216) are alternatives on one seam — how the RL manager explores — and were each written against the same base, never against each other. Merging them exposed three real defects that no author could have seen, all fixed in commit `9b33962`:

- `manager.get_action`: #213 gave `q_values` a head axis; #216's param-noise path reshaped its perturbed Q without one and handed both to `ParameterNoise.observe`, which broadcast-failed. The perturbed Q now collapses the head axis the same way `q_sel` collapses the reference. With #216's own K=1 both means are the identity, so that arm's behaviour is unchanged.
- `_StubManager` in `test_bootstrapped_manager.py`: #216 made `run_batch` call `begin_`/`end_behaviour_episode` on every rollout. The stub now implements them as the no-ops the real manager runs with `param_noise=None`.
- `_StubEnv` in the same file: #216's shape rows read `state['contribution_valid']`, which the real env carries and the stub did not.

These are edits I made, not any branch's. They are small and they are in one commit so they can be reviewed as a unit.

One seam is left deliberately unfixed: `ArtificalManager.behaviour_label` still reports `'eps-greedy'` for a bootstrap rollout, which is why the bootstrap case is named at the `rl_manager` call site instead. Making `behaviour_label` bootstrap-aware would be a new judgement that neither branch tested, so it is reported rather than invented.

### 3.4 #204 was in neither of the brief's lists

It has to be decided because #210, #211, #213, #215 and #216 are all based on it. It is in: it is a RESULT that changes no model, it is already the integration node that resolved #206 against #208, and its own additions are a new module (`linear_opponent.py`) and tests. Its finding — two of three seeds learned the inverse of the human policy, and a one-line rule beats the best learned seed — is a measurement, not a behaviour change.

---

## 4. Where I disagree with the brief

The brief listed **#202** and **#203** as likely excludes alongside #198 and #201, with the instruction to check their verdicts. Checked: neither is a failure. Both are additions-only RESULT branches with no source change at all, and both close a hypothesis rather than leaving one open.

- **#202**: the between-group lag-1 gap coefficient is 0.9098 in human games and 0.8973 in the honest copula-off simulation, a difference of 0.89 reseed sd, surviving seven alternative specifications. The drift defect is not a gain problem, and the real cause is named (smaller per-round innovation variance, 3.80 against 3.04).
- **#203**: "The contribution model does not under-respond to its group's level. The hypothesis is dead." Teacher-forced 0.2364 against human 0.2274; in the last third, where the defect lives, the two agree to 0.0008.

Under the stated criterion these are the same kind of thing as #211 and #215, which the brief lists as includes: a recorded null result that stops the next person re-running a dead hypothesis, costing nothing behaviourally. They are in. This is the one place the table departs from the brief's prior, and it is the easiest thing here to revert — they arrive as two merges (`bbe5f43`, `e73d622`), nothing else merged depends on them, and neither touches `src/`.

`#198` and `#201` are out, as the brief expected: #198 is a true null already condensed onto `main`, and #201 was refuted at the screen.

---

## 5. Two branches moved during the integration

`auto/rl-manager-bootstrapped-dqn` (#213) and `auto/rl-manager-param-noise` (#216) are the two arms still training, and both pushed new commits while `dev` was being built — #213's runs finished ("the inversion reproduces, and the heads disagree on the sign"). `dev` was re-fetched at the end and both were merged again at their then-current tips, so the branch carries their current state rather than a stale snapshot. Anything they push after this merge is not on `dev`. No remote run directory was touched: all testing used `~/repros/ai-runs/devint-a9837`.

## 6. PR #218 arrived after the cut and is deliberately not on `dev`

`auto/rl-manager-reward-targeting` (#218) was opened while this integration was running — it is the third live thing the brief named, the simulation in `~/repros/ai-runs/rl-reward-targeting`. It brings the open count to 75; the table above covers the 74 that existed at the cut.

Assessed but not merged. It is based on `auto/rl-manager-two-worlds` and contains #210, both of which are on `dev`, so it would merge as its own log, two analysis scripts, two configs and its plots. Its verdict — "the reward is a real defect and not the explanation: paying per capita moves targeting -0.25/-0.00/-0.22 when -1.1 is needed, and the managers stay inverted" — is a RESULT of exactly the kind #211 and #215 are in for, and it concludes the arm #210 launched.

**Recommendation: merge it into `dev` next.** It is not merged here because it landed after the branch was assembled and tested, and merging a branch whose run was still going when the brief was written is precisely the kind of unannounced decision the brief asked me not to make. It should be a clean merge and needs one Raven run to confirm; it does modify `src/aimanager/manager/environment.py` and `test_manager_reward.py`, though those edits appear to be #210's, which `dev` already has.

## 7. Known lint debt, not introduced here

`src/aimanager/rl_manager.py` is not `black==25.11.0` clean. Verified pre-existing: the same three hunks fail on `origin/auto/rl-manager-two-worlds` and `origin/auto/rl-manager-annealed-local` in isolation, and none of them is a line written for this integration. It is left alone rather than reformatted, so the diff stays reviewable — but the pre-commit hook runs `black` on `src/`, so the next commit touching that file will reformat it. `flake8 --max-line-length=88 --extend-ignore=E203,W503` is clean on every file edited here.
