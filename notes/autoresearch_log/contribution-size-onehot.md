# Autoresearch log: contribution — own-group size as `1/n` on the gaussian_mlp_v2 group-copula contributor

Branch `auto/contribution-size-onehot` (worktree
`.claude/worktrees/contribution-size-onehot`), created from
`origin/auto/switch-exodus-k-onehot` at `dd35443` — the head of the
maintainer-designated parent PR #174, per §9 "Building on a `[SUCCESS]` PR".
The PR opens with `--base auto/switch-exodus-k-onehot`. Remote isolation dir
`~/autoresearch/contribution-size-onehot`.

**Naming caveat, stated first.** The slug was assigned before the encoding
was chosen and reads "onehot"; the measurements below select the scalar
`1/n` instead (Declaration, "Encoding"). The slug is kept for the branch,
worktree, remote dir and this file, as assigned; configs, bundles and sim
output dirs are named by their content (`inv_size`) so a later reader is not
told an artifact is one-hot when it is not. Renaming the slug before step 1
is two commands and the orchestrator may do so.

## 1. Declaration

- **Slot:** contribution.
- **Parent PR:** **#174** `[SUCCESS] switch: one-hot group-size encoding in
  the joint exodus head` (`auto/switch-exodus-k-onehot`, stacked on #172 on
  #170 on #167). Its log is `notes/autoresearch_log/switch-exodus-k-onehot.md`;
  its post-verdict notes 20-21 locate the late SC rise and the singleton
  exodus residual in the *contribution* slot's inverted size -> per-capita
  common-good map and name this experiment as the successor; the
  maintainer's review comment 5527143609 on #174 is the assignment.
- **Base model:** the parent stack's contributor, PR #170's candidate
  `artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib`
  (sha256 `da42031ab0ca5bc2ea355036f2e07f5dd37b369d49bf0bf6999c9ebe68d0ea7c`)
  — the gaussian_mlp_v2 trunk
  (`artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib`, sha256
  `2f0b02e2588dbd8b2c4860ca3918d670095a6eb32851bcec931392c2d6a02e75`:
  2-layer heteroscedastic Gaussian MLP, hidden 8, wd 3e-4, lr 0.01, 1000
  epochs, seed 38381, 7 features `prev_contribution, prev_punishment,
  prev_contribution_mean_group, prev_punishment_mean_group,
  prev_win_contribution_mean_group, switched_last_choice,
  rounds_since_switch`, trained by
  `configs/training/baselines/contribution/gaussian_mlp_v2.yml` on the
  single-copy 40-episode train split) **plus** #170's stamped group copula
  (`copula_rho_p = 0.04378520865574197`, `copula_rho_t = 0.0`, censored
  pairwise MLE). The base is "trunk + stamped copula" and so is the
  candidate: the rho is model-conditional (teacher-forced marginals of the
  bundle being stamped), so it is re-estimated on the retrained trunk with
  #170's own estimator and stamper — the recipe's derived step, not a
  second change (PR #173 ruled the same way for the GNN stamp).
  **This slot's base model is a locally-trained joblib bundle sampled
  through `LinearAHAdapter`, not a GNN**: no Raven training, no
  `GraphNetwork.save` pickle, and PR #173's `make_contribution_copula_artifact.py`
  fix is irrelevant here (that is the GNN stamper; this lineage uses
  `scripts/baselines/stamp_contribution_group_copula.py`).
- **Evaluation stack (§3, parent rule of §9):** the parent's candidate config
  `configs/simulation/manager_testing/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
  — this contributor x the one-hot-k joint-exodus GNN switch
  (`artifacts/artificial_humans/switch_exodus_k_onehot/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`,
  sha256 `28dd4b40…c820d`) x PR #160 severity-copula multinomial punisher
  (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`,
  sha256 `9e3cf677…8cc2f`), `valid_model`
  `raven_script_22/model/rnn_False__dataset_full.pt`, single pairing
  `lin_multinomial_copula_self`, seed 42, 100 episodes, 24 rounds,
  `switch_every: 4`, `save_per_round: true` — with **only
  `contribution_model` swapped** (plus `output_dir` / `figure_name`).
- **Baseline for BOTH §2 gates**, read at full precision from
  `plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch/evaluation/scores.csv`
  in this worktree (21 rows, run `ah group_switching managed by
  lin_multinomial_copula_self`, 500/500 repeats, seed 42; the parent's
  `per_round.parquet` hashes
  `3cb8b3d72784afe09464e0048d27d7cb05f7a3b157780f296e99d168407fef3f`, equal
  to parent note 14). Every number the assignment quotes is reproduced by
  the file:

  | row | score | band | numerator | denominator |
  |---|---|---|---|---|
  | **CA** | **1.60900781342564514** | 1-2 | 1.67948943041112031 | 1.04380439696897231 |
  | CB | 0.93551251815132475 | <= 1 | | |
  | CC | 1.03555925747241551 | 1-2 | 1.03545848762063741 | 0.99990269040515922 |
  | CD | 1.11566724966523689 | 1-2 | 1.15853083757623887 | 1.03841968823935971 |
  | CE | 0.94741373065818557 | <= 1 | 1.37914893207194389 | 1.45569869576813504 |
  | CF | 1.34465307121114108 | 1-2 | 0.05621951948267670 | 0.04180968361752910 |
  | CG | 2.07926361874462717 | 2-5 | 0.05499377725668380 | 0.02644867960027440 |
  | SA | 0.81042746214736638 | <= 1 | 0.01713725315839310 | 0.02114594329390690 |
  | SB | 0.89051418259664639 | <= 1 | | |
  | SC | 0.97984865931896636 | <= 1 | 0.19060799999999931 | 0.19452799999999940 |
  | PA | 0.61906578987016991 | <= 1 | | |
  | PB | 0.96043898013168905 | <= 1 | | |
  | PC | 0.88830186447530723 | <= 1 | | |
  | PD | 0.86539523171815413 | <= 1 | | |
  | **RCA** | **3.50675152179351013** | 2-5 | 0.98186619820209642 | 0.27999309107020109 |
  | RCB | 1.91031928873782642 | 1-2 | | |
  | RCC | 1.08799671537859366 | 1-2 | | |
  | RCD | 0.73161386785466964 | <= 1 | | |
  | RSA | 1.31723440984737872 | 1-2 | | |
  | RPA | 1.31408428855398229 | 1-2 | | |
  | RPB | 0.72721365958192941 | <= 1 | | |

  mean **1.2226801514921317**, rows <= 1 **11/21** (context, not a
  criterion), **gate-2 ceiling 1.344948166641345** (= 1.1 x the mean in
  IEEE double; the assignment's `1.3449481666413449` is the same value
  printed to one more digit). Diagnostic `std_diff` for CA:
  −1.59345436047741806 (the sim's participant means are under-dispersed).

- **Target rows — adopted from the maintainer, not swapped.** Primary
  **RCA 3.50675152179351013** (`2-5`), needs `< 2`: the 500-repeat
  numerator must fall from 0.98186619820209642 below
  **0.55998618214040218** (2 x the denominator), −43%. Second candidate
  **CA 1.60900781342564514** (`1-2`), needs `<= 1`: numerator
  1.67948943041112031 -> at most 1.04380439696897231, −38%. **Pre-registered
  reading of both, from measurement (next bullets): neither is expected to
  cross.** Pre-declaration binds; no other row is claimed, and a band
  upgrade elsewhere (CG in particular — see the guards) is collateral.
- **Gate 2:** 21-row mean must stay <= **1.344948166641345** (0.122 of
  headroom above the parent's 1.2226801514921317).

### Why RCA cannot plausibly cross from a size feature (measured)

RCA is the EMD of `dc = c_{t+1} − c_t` within each of four round types,
weighted by the human type frequencies (`metrics.py::rca`,
`_round_types`). Decomposed on the human single-copy data and the parent's
parquet (`convert.load_human` / `load_sim`, read-only):

| round type | human weight | n_h | n_s | EMD(h, s) | w x EMD | human P(dc=0) | sim P(dc=0) | human dc mean | sim dc mean |
|---|---|---|---|---|---|---|---|---|---|
| no_switch_allowed | 0.7912 | 6902 | 14400 | 0.9065 | **0.7172** | 0.443 | 0.172 | −0.13 | +0.01 |
| stayed_comp_changed | 0.1263 | 1102 | 2376 | 1.0920 | 0.1379 | 0.496 | 0.175 | +0.08 | +0.01 |
| switched | 0.0617 | 538 | 1160 | 1.2959 | 0.0799 | 0.255 | 0.105 | +1.87 | +0.95 |
| chose_to_stay | 0.0209 | 182 | 464 | 1.0590 | 0.0221 | 0.522 | 0.177 | +0.02 | −0.08 |

Weighted d = **0.9571230433**, equal to `metrics.csv`'s RCA `d`
(0.95712304330758757; the scored numerator 0.9819 carries 2.6% finite-sample
inflation on top). A group-size feature acts where the size *changes* —
the `switched` and `stayed_comp_changed` types. **Setting both of those EMDs
to zero** (a perfect fix) leaves d = **0.7393** -> score ~2.64 (~2.71 in
numerator space), still `2-5`. The band edge needs d < ~0.546. The deficit
is the **exact-repeat spike in ordinary rounds**: human P(dc=0) 0.443
against the sim's 0.172, human kurtosis 8.2 against 3.3, and it is the same
at every group size (per-n EMD of no-switch dc: 0.93 / 1.00 / 0.98 / 1.11 /
0.87 / 0.88 / 0.78 at n = 2..8; only n = 1 is worse at 2.46, and n = 1 is
2.3% of the type). A Gaussian emission with sigma ~3.4 puts P(|N| < 0.5) ~
0.15 on dc = 0 whatever mu does — PR #156's "the emission owns RCA",
reconfirmed on this stack. Fixing n = 1 alone saves ~0.036 of d; fixing
the `switched` type's mean shift on becoming alone (human dc +1.50 on
`-> alone`, sim −0.36; human −1.02 on `from alone`, sim +0.15) is inside
the 0.08 that type carries in total. **Expected RCA after this change:
~3.2-3.4, within band.**

### Why CA cannot plausibly cross (measured)

CA is the EMD of participant mean contributions. Human SD of participant
means 5.07, sim 3.48 (std_diff −1.59). Regressing each participant's mean
on their mean experienced `1/n`: human slope −2.24, **R² 0.001**, implied
SD from size exposure 0.19; sim slope −8.38, R² 0.032 (the inverted map,
adding 0.62 of wrong-signed spread). Size exposure explains nothing of the
human between-participant spread; correcting the sim's inverted slope
removes spread rather than adding it. The static counterfactual below moves
CA's `d` 1.4796 -> 1.53 (worse). **Expected CA: ~1.6-1.7, within band.**

### The behavioral finding this experiment acts on (measured)

Contribution by own group size, rounds >= 4, valid rows:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| human mean (SE) | 10.54 (0.54) | 9.66 (0.29) | 8.82 (0.26) | 10.55 (0.26) | 10.06 (0.19) | 9.82 (0.16) | 9.75 (0.15) | 8.22 (0.15) |
| human sd / P(0) / P(20) | 8.01 / .23 / .31 | 6.69 / .10 / .17 | 6.81 / .13 / .16 | 7.14 / .14 / .20 | 6.56 / .09 / .15 | 6.43 / .09 / .15 | 6.10 / .08 / .13 | 5.17 / .07 / .07 |
| parent sim mean (SE) | 7.29 (0.27) | 9.60 (0.17) | 9.31 (0.16) | 9.66 (0.15) | 9.45 (0.12) | 10.41 (0.10) | 10.93 (0.10) | 11.23 (0.10) |
| sim sd / P(0) / P(20) | 5.86 / .16 / .05 | 5.76 / .08 / .06 | 5.64 / .07 / .05 | 5.63 / .06 / .07 | 5.39 / .05 / .05 | 5.60 / .05 / .08 | 5.61 / .04 / .10 | 5.60 / .04 / .09 |

OLS slope on n: human **−0.1514**, sim **+0.4019** (the assignment's
−0.151 / +0.402). Per-capita common good by n (rounds >= 4): human 15.6 /
13.7 / 12.8 / 15.5 / 14.9 / 14.4 / 14.2 / 11.8 (slope −0.150), sim 9.8 /
13.9 / 13.4 / 14.0 / 13.6 / 15.4 / 16.4 / 16.2 (slope **+0.739**) — the
inverted map parent note 20 found. Within-player response to a size change
at decision rows: human dc regressed on Δ(1/n) has slope **+1.85** (n =
1560), the sim **−0.89**; human dc on `-> alone` +1.50 (n 52), sim −0.36
(n 118); human `-> eight` +0.88, sim +0.60.

**Where the incumbent actually fails, conditional on its own features
(teacher-forced on the 40-episode train split, rounds >= 4, y − mu by n):**

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| rows | 190 | 432 | 573 | 657 | 950 | 1304 | 1372 | 745 |
| residual | **+2.599** | **+0.411** | −0.167 | −0.163 | −0.006 | +0.044 | −0.031 | −0.081 |
| SE | 0.335 | 0.213 | 0.187 | 0.147 | 0.118 | 0.090 | 0.084 | 0.120 |
| sigma(x) mean | 5.07 | 3.78 | 3.77 | 3.24 | 3.35 | 3.20 | 3.02 | 3.24 |

The incumbent under-predicts lone players by **2.6 points at 7.8 SE** and
pairs by 0.4 at 1.9 SE, and is unbiased at n >= 3 — including n = 8 (mu
7.40 vs y 7.32), so the sim's merged-eight excess (11.2 vs 8.2) is
free-running composition/drift, not a conditional the feature could fix.
Unlike PR #173's GNN, whose feature moved nothing resolvable at the
teacher-forced marginal, the defect here is large and resolvable *before
anything is trained*.

- **Hypothesis (behavioral, one sentence):** a contributed point returns
  `1.6 / n` to the contributor — 1.6 alone, 0.2 in a merged eight — and
  humans hold that payoff right (lone players contribute 10.5, merged
  eights 8.2, within-player dc +1.5 on becoming alone) where the
  contributor, which has no own-group-size input at all, has it backwards
  (7.3 alone, 11.2 merged, dc −0.4 on becoming alone); giving it the
  marginal per-capita return factor as a feature should restore the
  singleton level (the measured 2.6-point conditional gap) and with it the
  sign of the size -> per-capita-common-good map the switch trunk reads.
  The rows this *should* move per the maintainer are RCA (via the
  `switched` / `stayed_comp_changed` types) and CA; the measurement above
  says those moves stay inside their bands.

### Encoding: `1/n`, chosen by measurement, not by preference

Both allowed encodings were fitted through the incumbent's exact procedure
(4-fold CV by pair, seed 38381, hidden 8, wd 3e-4, lr 0.01, 1000 epochs,
`StandardScaler` per fold, the incumbent's 7 features plus the size term;
the incumbent arm reproduces its recorded CV NLL 2.692550840548696 to the
digit, so the comparison is apples to apples):

| arm | features | CV NLL (folds) | CV binned CE | TEST NLL | TEST binned CE | train residual at n=1 (SE 0.33) |
|---|---|---|---|---|---|---|
| incumbent | 7 | 2.69255 (2.6488 / 2.7114 / 2.5485 / 2.8615) | 2.39917 | 2.64112 | 2.31011 | +2.599 |
| one-hot n in {1..8} | 15 | **2.75033** (2.7485 / 2.7770 / 2.5327 / 2.9431) | 2.45721 | **2.99730** | 2.31364 | −0.192 |
| `1/n` | 8 | **2.70171** (2.6543 / 2.7522 / 2.6032 / 2.7971) | 2.41609 | 2.65039 | **2.29761** | +0.032 |

The one-hot costs **+2.1% held-out NLL** (worse on 3 of 4 folds) and its
**test NLL rises 13% while the binned CE is flat** — the sigma-collapse
tail pathology that killed PR #151 (a few rows with a very sharp sigma
and a large error), from eight sparse standardised columns (the n = 1
column standardises to +6.2 for a singleton) into a hidden layer of 8 on
7457 rows. The scalar `1/n` is essentially free (+0.3% CV NLL, test CE
*better* than the incumbent), and it captures the whole resolvable effect:
the n = 1 residual goes +2.599 -> +0.032, n = 2 +0.411 -> +0.196, n >= 3
within 2.4 SE of zero (test split: n = 1 residual +3.14 -> +0.34). The
tanh layer bends the scalar into the step the data shows, so the argument
that "1/n cannot represent a non-monotone profile" (PR #173's successor
note, and my own first reading) does not hold for this model — the
*marginal* human profile is non-monotone (a + b/n rejected at chi² 105 on
6 df), but the *conditional* residual the model has to close is a
singleton step, and `1/n` closes it. §5's tie-break points the same way:
one parameter block, not eight. **Planned feature: `inv_group_size = 1 /
group_size`**, where `group_size` is the current-round head-count of the
agent's own group already computed by `build_feature_pool`
(`scripts/baselines/handcrafted_grid.py:276-279`), legal for the
contribution target by `notes/baseline_feature_defs.md` ("membership
itself resolves before contributing, so current membership-derived
features (sizes, tenure counters) are legal for both targets") and not in
`CURRENT_VALUED`. One-hot is **not** run as a second arm.

- **Planned change (one change, contribution slot only):** add
  `inv_group_size` to the feature pool; retrain the gaussian_mlp_v2 trunk
  with the incumbent's 7 features + `inv_group_size` (8 features) at the
  incumbent's exact setting — **no grid, no arm selection by score**;
  re-estimate `rho_total` on the retrained trunk with #170's estimator and
  stamp it with #170's stamper (`rho_p = rho_total`, `rho_t = 0`, #165's
  stop-gate on a CI including 0 carried verbatim); swap the stamped bundle
  into the parent's config. Everything else is byte-identical to the
  parent: switch model, punisher, valid model, seed 42, 100 episodes, 24
  rounds, single pairing, the flip/single-copy data conventions.
- **Guards (declared, non-gating, with the reading fixed in advance).**
  A static counterfactual on the parent's parquet — shift every n = 1
  contribution by the measured +2.6 and every n = 2 by +0.4 (the level-only
  fix a Gaussian head delivers), clipped and rounded — predicts:

  | row | baseline score | static prediction | direction |
  |---|---|---|---|
  | **CG** 2.07926361874462717 (`2-5`) | d 0.05466 (ratio 0.7934 vs human 0.8480) | ratio 0.7746, score **~2.77** | worse |
  | **CC** 1.03555925747241551 (`1-2`) | d 0.7302 | d 0.9237, score ~1.31 | worse |
  | **CE** 0.94741373065818557 (`<= 1`) | d 1.0257 | d 1.2452, score ~1.15 | worse, leaves `<= 1` |
  | CD 1.11566724966523689 | d 0.9487 | d 1.0021 | slightly worse |
  | CA 1.60900781342564514 | d 1.4796 | d 1.5268 | slightly worse |
  | CF 1.34465307121114108 | d 0.05279 | d 0.05393 | flat |

  The sim's too-low singletons were *accidentally* adding group-mean spread
  that mimics the human singleton's bimodality; raising their level with an
  unchanged Gaussian shape removes it. **CG is therefore a guard expected
  to move against us, not a target**: it sits 0.079 over the `1-2` edge
  with an episode-bootstrap sd of **0.0107 in ratio = 0.40 score units**,
  so a crossing in either direction is a coin flip and would not be
  claimed. (For the record: resampling the sim's n = 1 cells from the
  *human* n = 1 distribution — bimodal, sd 8.0 — takes CG to 0.50 and CE
  to 0.76 on the same parquet. The scored rows reward the singleton's
  *shape*, which a Gaussian emission cannot produce; that is a successor's
  emission question, not this experiment's.) Summed, the static predictions
  cost ~+1.3 score points over the 21 rows, ~+0.06 on the mean -> ~1.28,
  inside the 1.345 ceiling; **gate 2 is expected to pass**.
  - **SA 0.81042746214736638 (`<= 1`) and SB 0.89051418259664639 (`<= 1`):**
    the propagation channel PR #173 identified — the switch trunk's only
    group-quality input is per-capita common good, a function of
    contributions. A singleton at 10.5 instead of 7.3 reads per-capita CG
    ~16 instead of 9.8 and leaves less often (the trunk's own rule, parent
    note 20), so switch rates fall; the candidate's per-round rates already
    sit under the human's at rounds 3, 7, 19. SA/SB may leave `<= 1`. Not
    claimed in either direction.
  - **SC 0.97984865931896636 (`<= 1`, margin 0.02):** the parent's target,
    bought thinly (single-draw noise 0.1-0.2, parent note 6). The
    maintainer's knock-on says the corrected size -> CG map should *lower*
    the late SC drift; but SC's band cannot improve, and a regression past
    1.0 would be this experiment's most consequential cost. Reported.
  - **RCD 0.73161386785466964, RCB 1.91031928873782642, RSA
    1.31723440984737872:** inherited guards; no mechanism claim, expected
    to wobble with the retrain (R-row retrain wobble ~0.2-0.36, PR #173).
- **Pre-registered expectation, stated plainly:** **`[FAIL]` on gate 1.**
  Both declared targets are measured out of reach of a size feature; the
  guards predict a C-block cost; gate 2 should hold. The experiment's
  value, if run, is (i) a resolvable, pre-measured conditional defect
  (the 2.6-point singleton gap) fixed at zero likelihood cost, (ii) the
  first closed-loop measurement of whether restoring the singleton's
  payoff-correct level also restores the singleton's *staying* (the
  maintainer's knock-on: P(full exodus | k = 1) 0.371 vs human 0.161, late
  SC anchors 6.36 / 6.50 vs 5.92 / 5.84, per-capita CG gap small − big at
  |Δn| = 6 of −6.5 vs +0.6), which no static counterfactual can answer, and
  (iii) a measured test of the static cost prediction. That is the honest
  claim. The orchestrator and maintainer should decide whether that is
  worth two 3-minute sims; the plan below runs them.
- **Pre-registered falsifiers and unsoundness criteria (numbers fixed
  now):**
  - **F1 — learned (pre-sim, from the committed pipeline, not the scratch
    fit).** The refit trunk's teacher-forced train residual at n = 1 must
    lie within **±0.7** of zero (2 SE; from +2.599) and its 4-fold CV NLL
    must not exceed **2.7125** (incumbent + 0.02; the scratch fit gives
    2.70171 — the one-hot's 2.75033 would fail this). Test binned CE
    reported against 2.3101098745482482. **F1 failing is a stop**, not a
    "run anyway": the scratch measurement says the pipeline passes it, so a
    failure means the committed feature or config differs from what was
    measured — a bug to fix before any simulation.
  - **F2 — #165/#170's copula stop-gate, verbatim:** if the re-estimated
    `rho_total`'s 200-resample cluster-bootstrap CI includes 0, the
    experiment ends as a calibration-only `[FAIL]` (no stamp, no sim).
    Expected: near 0.0438 (PR #173 measured +8% on a retrained trunk with
    the same 7457 rows / 15090 pairs; those two counts must reproduce).
    Round-trip max |bias| <= 0.02 or stop (implementation bug).
  - **F3 — the mechanism did what it claims (closed loop, from the
    candidate's parquet, rounds >= 4).** (a) mean contribution at n = 1
    rises from **7.29 to >= 8.9** (half the gap to the human 10.54); (b)
    the per-capita common-good slope on n falls from **+0.739 to <= +0.30**
    (human −0.150) and singleton per-capita CG rises from **9.83 to >=
    12.7** (half the gap to 15.6); (c) dc on `-> alone` at decision rows
    turns positive (from −0.36; human +1.50). If a declared row crosses a
    band while F3 fails, the result is mechanistically unsound and the log
    says so whatever the title.
  - **F4 — the cost prediction.** CG, CC and CE are predicted to move
    *against* us by the static table. If instead they improve, the closed
    loop did something the static counterfactual cannot see (composition:
    singletons that stay), and that is recorded as the finding — not
    claimed as a success, since none of the three is declared.
  - **Knock-on, reported, never claimed:** P(full exodus | k) on complete
    pairs (parent 33/89 = 0.371 at k = 1, 20/112 = 0.179 at k = 2; human
    0.161 / 0.147), the full-exodus cell share (parent 0.1311, human
    0.1079), SC anchors at rounds 4/8/12/16/20, P(L = 8) by four-round
    block, and the per-capita CG gap by |Δn|.
- **Iteration budget (§5).** The contribution slot's base model trains
  **locally in 0.71 s per fit** (measured today at the incumbent setting on
  7457 rows; 0.70 s with the extra columns), so the 3x ceiling is ~2 s per
  fit and the whole retrain — 4 CV folds + full refit + test — is ~6 fits.
  The copula estimator is seconds plus a low-minute bootstrap and
  round-trip (PR #170 ran it locally). Cluster wall-clock: two simulations
  of ~2-3 min each, nothing else. The assignment's ask to read the
  contribution model's training time off Raven `train-ah` logs does not
  apply: this slot's model is not a GNN and never trained on Raven.
- **Legality and frozen surface.** One new feature, `inv_group_size`,
  derived from the current-round head-count the pool already computes;
  legal by the documented rule (membership resolves before contributing),
  not in `CURRENT_VALUED`, not keyed to any metric's bin or stratum (RPB's
  size bins belong to the punisher row; RCA has no size stratum). No seed,
  episode count, scoring parameter or protocol field changes. Nothing under
  `src/aimanager/evaluation_suite/`, `notes/evaluation_metric_defs.md`,
  `notes/eval_scoring_schema.md` or `experiments/` is touched; the
  diagnostics import `convert.load_human` / `load_sim` and `metrics`
  read-only from scratchpad scripts. `notes/baseline_feature_defs.md` (not
  frozen) gains one line for the new feature. No other branch's log file
  is written. The estimator/stamper edits of step 5 are CLI
  parametrisation with defaults equal to today's constants — no numeric
  path changes.

## 2. Plan

Steps for the orchestrator to validate (§2 targets, §5 legality, §8 frozen
surface) and tag. Paths are relative to the worktree
`.claude/worktrees/contribution-size-onehot`. Local Python is the main
checkout's venv with the worktree's source first on the path
(`PYTHONPATH=$PWD/src /Users/ertuerkan/Desktop/algorithmic-institutions/.venv/bin/python …`;
`uv sync` cannot run in the worktree, parent note 9); `mkdir -p
data/baselines` first (gitignored CV output dir). `AI_REMOTE_DIR='~/autoresearch/contribution-size-onehot'`
on every `simulate_cluster.sh` / `fetch_cluster.sh` / `remote_test.sh`
call, and `squeue -u certuer` over a live SSH tunnel before any syncing
call. Lint (`black`, `flake8 --max-line-length=88 --extend-ignore=E203,W503`
on `src/`) once per step before staging.

1. **The `inv_group_size` pool feature** — `scripts/baselines/handcrafted_grid.py`,
   `build_feature_pool` (existing), directly after `f["group_size_delta"] =
   own_cur - oth_cur` (line ~279): `f["inv_group_size"] = 1.0 / own_cur`,
   preceded by `assert (own_cur >= 1).all()` (the agent is a recorded member
   of its own group, so the head-count is never 0 — the same `recorded`
   convention `group_size` uses; in the sim every agent is recorded). Not
   added to `CURRENT_VALUED`; `validate_feature_legality` and the adapter's
   load-time assert therefore accept it for the contribution target. One
   line under "Current -- Group" in `notes/baseline_feature_defs.md`:
   `inv_group_size: 1 / group_size — the marginal per-capita return factor
   (a contributed point returns 1.6 / n to the contributor)`. Tests in
   `tests/baselines/test_baseline_features.py` (existing): add
   `"inv_group_size"` to the `C2` list (so `ALL_FEATURES` and the pipeline
   parity test cover it) and to `CONTRIB_SAFE` (so the contribution
   adapter's parity test covers it); in `build_reference`, after
   `ref["ref_group_size"] = col("size_grp")`, add `ref["ref_inv_group_size"]
   = 1.0 / ref["ref_group_size"]`. The three parametrised parity tests
   (`test_feature_matches_pipeline`, `test_adapter_matches_reference`,
   `test_switch_adapter_matches_reference`) then bind training pool, sim
   adapter and switch adapter to one reference for the new feature; the
   two-implementation hazard PR #173 had does not exist here because
   `LinearAHAdapter` calls the same `build_feature_pool`. Run
   `pytest tests/baselines -q` (parent-of-parent's figure: 280 passed);
   `prepare_data`'s `sorted(pool)` shifts column indices by one key —
   nothing persisted depends on them (bundles store feature *names*).

2. **Training config** — new
   `configs/training/baselines/contribution/gaussian_mlp_v2_inv_size.yml`:
   `data` block byte-identical to `gaussian_mlp_v2.yml` (train file,
   `exclude_flipped: True`, `mask: contribution_valid`, `switch_every: 4`,
   `model: gaussian_mlp`); `cv` identical except `output:
   data/baselines/gaussian_mlp_v2_inv_size_cv.csv`; `setting` pinned to
   **scalars** `hidden: 8`, `weight_decay: 0.0003`, `lr: 0.01`, `epochs:
   1000` (the incumbent's, no grid); `blocks` reduced to one block
   `B_declared` with one set — the incumbent's seven
   `[prev_contribution, prev_punishment, prev_contribution_mean_group,
   prev_punishment_mean_group, prev_win_contribution_mean_group,
   switched_last_choice, rounds_since_switch]` plus `inv_group_size`.
   `enumerate_feature_sets` then yields exactly two rows (floor, the set).
   Header comment: the Declaration's one-sentence rationale and the
   encoding table's numbers, so the file explains itself.

3. **Fit, save, F1** — `run_baseline_cv.py configs/training/baselines/contribution/gaussian_mlp_v2_inv_size.yml`
   (expected rank-1 row: the 8-feature set, CV NLL **2.70171**, CE
   **2.41609**, folds 2.6543 / 2.7522 / 2.6032 / 2.7971; floor rank 2), then
   `inspect_best_model.py data/baselines/gaussian_mlp_v2_inv_size_cv.csv
   --config <step-2 config> --save-best --name
   contribution_gaussian_mlp_v2_inv_size_best.joblib` (expected TEST NLL
   2.65039, TEST binned CE 2.29761 against the incumbent's 2.6411174392098506
   / 2.3101098745482482; `sigma_mean` reported). Then the F1 read: new file
   `scripts/baselines/size_profile_preflight.py` (local, CPU torch;
   `--config`, `--candidate`, `--incumbent` like `gaussian_mlp_preflight.py`)
   printing, for both bundles on the train and test splits (rounds >= 4),
   the teacher-forced `y − mu` by `group_size` with SE, mean `mu` and mean
   `sigma(x)` by size, against the human means of the Declaration's table.
   **F1: candidate train residual at n = 1 within ±0.7 and CV NLL <=
   2.7125, else stop and debug.** Also run the existing
   `gaussian_mlp_preflight.py --candidate <new> --incumbent <old>` and
   record its conformity `a`, `b`, `a + b` (parent-of-parent's diagnostic;
   incumbent 0.7316 / 0.2015 / 0.9331) and exact-repeat mass. Commit the
   bundle (plain ~4 KB binary, not LFS — `.gitattributes` tracks only
   `*.csv`, `*.parquet`, `*.pt`) and its sha256 in the Notes.

4. **Parametrise #170's estimator and stamper (CLI only, defaults =
   today's constants)** — `scripts/baselines/contribution_gmlp_copula_rho.py`,
   `main` (existing): add `--bundle` (default the current `BUNDLE_PATH`),
   `--config` (default `TRAIN_CFG`) and `--out` (default `OUT_JSON`), and
   use them everywhere the module constants are read in `main` and in the
   `params` dict (`base_bundle`, `base_bundle_sha256`, the two
   `relative_to(ROOT)` prints). `REF_N_PAIRS_WITHIN = 15090` and every
   estimator, bootstrap, round-trip and power-arm setting stay verbatim.
   `scripts/baselines/stamp_contribution_group_copula.py` (existing): add
   `--base` (default `BASE_PATH`), `--params` (default `PARAMS_PATH`),
   `--out` (default `OUT`); `load_inputs` asserts the on-disk base sha256
   equals the sidecar's `base_bundle_sha256` (the provenance check that
   matters) and additionally equals `EXPECTED_BASE_SHA256` only when running
   with the default `--base`; in `check_adapter_equivalence` replace the
   literal `assert rho_p == 0.04378520865574197` with `== float(params["rho_total"])`
   (thread `params` through). The `NEW_KEYS` manifest, the identical-object
   check, the reload bit-identity check and `build_new_bundle` are
   untouched. Add to `tests/baselines/test_contribution_group_copula.py`
   (existing) one test that the two scripts' argparse defaults equal the
   module constants — the guarantee that the parent's invocation is
   unchanged. **Do not run either script with defaults**: that would
   overwrite the parent's committed sidecar/bundle.

5. **Calibrate rho on the new trunk (F2 stop-gate)** — locally,
   `contribution_gmlp_copula_rho.py --bundle
   artifacts/baselines/contribution_gaussian_mlp_v2_inv_size_best.joblib
   --config <step-2 config> --out
   artifacts/baselines/contribution_gaussian_mlp_v2_inv_size_group_copula.params.json
   --write-params`. Record: `rho_total`, its 200-resample CI and SE, the
   pairwise LR, `rho_lag1` and CI (provenance only), rows **7457**, pairs
   **15090** (both must reproduce; a different count means the row builder
   changed), censored share, round-trip verdict (tol 0.02) and the power
   arm. **STOP-GATE (F2): CI including 0 -> calibration-only `[FAIL]`, no
   stamp, no sim.** Expected `rho_total` near 0.0438. Commit the sidecar.

6. **Stamp** — `stamp_contribution_group_copula.py --base <step-3 trunk>
   --params <step-5 sidecar> --out
   artifacts/baselines/contribution_gaussian_mlp_v2_inv_size_group_copula.joblib`.
   Its six checks must all print PASS (identical objects, exact manifest,
   `predict` / `predict_std` bit-identical after reload, `sample=False`
   adapter equivalence over the fixed 6-round switching sequence, rho
   read-back, sha256). Record the bundle sha256 (re-checked on Raven in
   step 8). Commit the bundle.

7. **Sim configs, control and candidate** — two new files under
   `configs/simulation/manager_testing/`, each a byte-copy of the parent's
   `23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`:
   (a) **control**
   `23_2g8a_sizeinvctl_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
   with exactly two edits (`output_dir`, `figure_name` -> the `sizeinvctl`
   name): the parent's own stack re-simulated on this branch's code, which
   must reproduce the parent's `per_round.parquet` **bit for bit**
   (`3cb8b3d7…fef3f`) — the proof that step 1's pool change is inert for a
   bundle that does not name the feature, and what licenses judging against
   the parent's committed `scores.csv`; (b) **candidate**
   `23_2g8a_sizeinv_self_gaussian_mlp_v2_inv_size_group_copula_contr_gnn_joint_exodus_k_onehot_switch.yml`
   with exactly three edits (`contribution_model` -> the step-6 bundle,
   `output_dir`, `figure_name`). Both parse under `evaluation_sweep.py`'s
   `_self_(\w+?)_contr_(\w+?)_switch$` (contr `gaussian_mlp_v2_group_copula`
   / `gaussian_mlp_v2_inv_size_group_copula`, switch
   `gnn_joint_exodus_k_onehot`). Verify by `diff` (2 / 3 changed lines; seed
   42, 100 episodes, 24 rounds, `switch_every 4`, single pairing,
   `save_per_round: true` byte-identical).

8. **Raven: control then candidate, isolated** — `squeue -u certuer`; one
   syncing call `AI_REMOTE_DIR='~/autoresearch/contribution-size-onehot'
   scripts/simulate_cluster.sh <control config>` (the sync ships
   `artifacts/` including the committed step-6 bundle; nothing exists
   remotely before this call, so the `rsync --delete` hazard PR #173 hit
   after a remote training run does not arise, and `plots/` is excluded
   from the sync so sim outputs are never deletion candidates), then
   `--no-sync` for the candidate; wait with one blocking `ssh raven 'while
   squeue -j <ids> -h | grep -q .; do sleep 30; done; sacct -j <ids>
   --format=JobID,State,ExitCode -n'`. From each log: the PROVENANCE line
   names the shared venv's interpreter and `aimanager.__file__` under
   `~/autoresearch/contribution-size-onehot/src/`; `algorithmic-institutions/src`
   absent; remote sha256 of the three slot artifacts (switch `28dd4b40…`,
   punisher `9e3cf677…` shared; contribution `da42031a…` for the control,
   step 6's hash for the candidate). **Activation check:** the candidate
   parquet must **differ** from `3cb8b3d7…`. One draw each, seed 42, no
   re-runs.

9. **Fetch, evaluate, diagnose** — `AI_REMOTE_DIR=… scripts/fetch_cluster.sh
   plots/simulation/<control dir>` and `<candidate dir>`; confirm the
   control's sha256 locally; `python -m aimanager evaluate <candidate
   config>`. Fill the Results row with RCA, CA, the mean, rows <= 1, and
   the guards CG / CC / CE / CD / CF / SA / SB / SC / RCD / RCB / RSA
   exactly as computed. Then the read-only diagnostic — an **uncommitted
   scratchpad script** importing `convert.load_human` / `load_sim` and
   `metrics.ResponseMetrics` only — on human, parent candidate and this
   candidate: F3 (a)-(c); the RCA decomposition by round type and by n
   against the Declaration's table; contribution and per-capita CG by n
   with slopes; the F4 rows against the static prediction; the knock-on
   set (P(full exodus | k) with cell counts on complete pairs, full-exodus
   cell share, SC anchors, P(L = 8) by block, per-capita CG gap by |Δn|).
   State F1-F4 outcomes in numbers. Commit sim outputs and evaluation only.

10. **[Orchestrator] Verdict, log, PR, clean-up** — §2 on the single
    evaluation, no second stage: `[SUCCESS]` iff RCA < 2.0 or CA <= 1.0,
    **and** mean <= 1.344948166641345; otherwise `[FAIL]`. Complete Results
    and Notes (the F1-F4 outcomes, the encoding table, the static-vs-realised
    cost, the knock-on numbers, collateral `+` / `-`). `gh pr create --base
    auto/switch-exodus-k-onehot`, body Hypothesis / Results / Collateral
    (§9.7), stating up front the pre-registered expectation of this
    Declaration and that the diff shows only this experiment's change over
    PR #174. Delete `~/autoresearch/contribution-size-onehot` on Raven when
    the PR closes, not at open.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|

## 4. Notes
