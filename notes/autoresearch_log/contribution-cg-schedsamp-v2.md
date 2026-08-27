# Autoresearch log: contribution-cg-schedsamp-v2

Resume of the `[ABANDONED]` `auto/contribution-cg-schedsamp` experiment
(abandoned 2026-08-13 by the maintainer as a cost/time call, before any arm
finished training; implementation complete and fully tested there), rebased
onto the `[SUCCESS]` parent PR #160 (`auto/punisher-severity-copula-v2`)
under the current single-stage, two-gate protocol. Maintainer-requested
addition: a cheap two-arm pilot (with / without scheduled sampling) that
must show a CG effect and yield a wall-clock estimate before the full-size
training runs are submitted; if the pilot does not separate, stop and ask
the maintainer.

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gnn`
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`)
- **Parent PR (§9):** #160, branch `auto/punisher-severity-copula-v2`.
  Branch and worktree created from its head (`b7dabfc`); the PR targets
  that branch so the diff shows only this experiment.
- **Evaluation stack (§3, inherited from the parent):**
  `gnn x gnn x lin_multinomial+copula`, sim config
  `configs/simulation/manager_testing/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`.
  Baseline = the parent's confirmed scores
  (`plots/simulation/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch/evaluation/scores.csv`):
  **CG 9.808514112722413**, RCD 2.941928428442498, RCA 2.0829074791966917,
  rows <= 1 **10/21**, mean **1.6879978841849728**.
- **Target rows:** **CG** 9.808514 (band > 5; gate 1 needs < 5) — the worst
  row of the entire score matrix, concordant across all GNN-contribution
  contexts. Secondary declared targets: **RCD** 2.941928 and **RCA**
  2.082907 (band 2-5; gate 1 satisfied by 1-2 or better) — the
  punishment-response rows are scored on free-running histories, so
  training under self-generated histories should sharpen the conditional
  response they measure.
- **Hypothesis:** the CG deficit is free-running state-tracking loss
  (exposure bias), not sampling structure: PR #149 measured the same
  model + independent sampler at spread ratio 0.784 teacher-forced on real
  histories (human 0.847) vs ~0.59 free-running (floor 0.583), with the
  within-round sampler worth only ~14% of the gap. Trained exclusively on
  human histories, the model mean-reverts when conditioned on its own
  slightly-off outputs and the drift compounds over 24 rounds. Training it
  under its own rollouts should teach it to hold a group norm through
  self-generated histories, recovering group-mean spread (CG) and the
  free-running punishment response (RCD/RCA).
- **Planned change:** the scheduled-sampling curriculum from
  `auto/contribution-cg-schedsamp`, restored unchanged (train.py steps
  2-4 + `test_scheduled_sampling.py`; 98 remote + 89 local tests were green
  there, p=0 reproduces the legacy path bit-identically at the input level,
  logits to ~1e-5): training unrolls episodes round by round; with
  per-agent probability p (0 below ramp_start, linear to p_max at
  ramp_end, hold after), `prev_contribution` at round t is replaced by the
  model's own detached sample from round t-1. Targets remain the true
  human actions; no architecture change, same feature set. p_max selected
  between arms by evaluation score (legal per §5).
- **Pilot (maintainer-requested, precedes the full runs):** two ~150-epoch
  arms — control p_max=0 and schedsamp p_max=0.5 with the ramp scaled
  proportionally — each simulated under the standard 23-family protocol
  and evaluated; compare CG and the free-running spread ratio, and measure
  per-epoch wall time of the unrolled path to price the 575-epoch runs.
  **Stop-and-ask gate:** if the pilot arms do not separate on CG, escalate
  to the maintainer instead of submitting the full-size jobs. The pilot's
  fold-loss trajectory in the hold phase also answers whether 575 epochs
  suffice for the real run or need extending.
- **Guardrail watch (not targets):** the marginal C block (CA/CB/CD/CF,
  all <= 1) — the failure mode that killed #154/#155/#156; and the known
  risk that held-out teacher-forced log-loss ticks up while free-running
  behavior improves (declared on the original branch; the evaluation suite
  is the judge, log-loss is a diagnostic only).

## 2. Plan

Validated by the orchestrator 2026-08-25 (targets per §2, legality per §5,
frozen surface per §8). Slug: `schedsamp_v2`. Orchestrator-approved planner
recommendations: pilot keeps `n_cross_val: 5` (nulling it skips the
held-out folds entirely and destroys the hold-phase loss trajectory the
pilot exists to produce) and sets `train_args.eval_period: 10` in both
pilot arms (reference 50 gives only 2 hold-phase eval points; shared value
keeps ctrl-vs-p50 fair).

- [x] 1. Cherry-pick the implementation unchanged from the local branch
      `auto/contribution-cg-schedsamp` (aec2fd3 train.py curriculum,
      7a34868 test_scheduled_sampling.py); verify
      `git diff auto/contribution-cg-schedsamp HEAD -- <the two files>`
      is empty.
- [x] 2. Tests: `squeue --me` PENDING/RUNNING check before any rsync;
      `scripts/remote_test.sh` (expect the 8 scheduled-sampling tests
      green — the old log's "7" was a miscount); local eval-suite +
      scripts + baselines suites green.
- [x] 3. Pilot training configs, copies of
      `configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml`
      changing only description/labels/output_dir/epochs(150)/
      eval_period(10)/scheduled_sampling:
      `auto_cg_schedsamp_v2_pilot_ctrl.yml` (no scheduled_sampling key) and
      `auto_cg_schedsamp_v2_pilot_p50.yml` (p_max 0.5, ramp 22->90;
      `labels.schedsamp: '0.50'` quoted — the filename depends on it).
      Same seed 38381, n_cross_val 5, flip-doubled 50ep data.
- [x] 4. Train both pilots on Raven (`train_cluster.sh ah`, second with
      `--no-sync`); poll; fetch artifacts + metrics; record per arm:
      sacct Elapsed and per-epoch time (Elapsed / (6 folds x 150)),
      realized substitution rate and p(e), held-out log_loss trajectory
      per cv_split.
- [x] 5. Pilot sim configs: byte-identical protocol copies of
      `23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`, only
      contribution_model/output_dir/figure_name changed:
      `23_2g8a_schedsamp_v2_pilot_ctrl_self_gnn_contr_gnn_switch.yml`
      (artifact `.../auto_cg_schedsamp_v2_pilot_ctrl/model/architecture_node+edge+rnn__dataset_50ep__epochs_150.pt`)
      and `23_2g8a_schedsamp_v2_pilot_p50_self_gnn_contr_gnn_switch.yml`
      (artifact `...__epochs_150__schedsamp_0.50.pt`); verify paths against
      the fetched filenames and the sweep DIR_PATTERN.
- [x] 6. Pilot sims on Raven (squeue check; confirm punisher joblib +
      switch + valid artifacts exist remotely); fetch; `python -m
      aimanager evaluate` both; report unrounded CG/RCD/RCA, rows <= 1,
      mean, plus the read-only `_spread_ratio` diagnostic (human, ctrl,
      p50, parent baseline).
- [x] 7. DECISION GATE (orchestrator; STOP POINT): (a) p50 CG separated
      below ctrl CG with spread ratio moving toward human — if not,
      STOP and escalate to the maintainer; (b) 575-epoch wall-clock
      estimate vs the 10 h template wall; (c) hold-phase fold-loss still
      descending? -> choose epochs E (default 575) for the full arms.
- [x] 8. Full arm configs `auto_cg_schedsamp_v2_p25.yml` /
      `auto_cg_schedsamp_v2_p50.yml`: epochs E, ramp 86->345, seed 38381,
      n_cross_val 5, quoted schedsamp labels, output_dirs
      `artifacts/artificial_humans/auto_cg_schedsamp_v2_p{25,50}`.
- [x] 9. Train both full arms on Raven; verify sacct State COMPLETED (not
      TIMEOUT); fetch; log fold-mean best held-out log_loss (diagnostic
      only), realized substitution rate, Elapsed.
- [x] 10. Full-arm sim configs
      `23_2g8a_schedsamp_v2_p{25,50}_self_gnn_contr_gnn_switch.yml`,
      same copy discipline, artifact paths verified against the fetched
      filenames.
- [x] 11. Simulate both on Raven; fetch; evaluate both; report all 21
      rows unrounded per arm + spread-ratio diagnostic + CA/CB/CD/CF
      guardrail block.
- [x] 12. Arm selection (orchestrator): better arm by evaluation score;
      ties to the lower p_max.
- [x] 13. Verdict per §2 vs the declared baseline: CG > 5 -> 2-5 or
      better, OR RCD/RCA 2-5 -> 1-2 or better, AND mean <
      1.6879978841849728. Complete the Results table unrounded.
- [ ] 14. Push; PR with `--base auto/punisher-severity-copula-v2`, title
      `[SUCCESS]`/`[FAIL]`, body Hypothesis / Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-25 | (baseline) parent stack, severity-copula punisher (PR #160) | — | CG 9.808514112722413, RCD 2.941928428442498, RCA 2.0829074791966917 | 10/21 | 1.6879978841849728 | baseline |
| 2026-08-25 | pilot ctrl: 150 epochs, no schedsamp (jobs 29612646 train / 29613554 sim) | pilot | CG 9.592575, RCD 3.484160, RCA 3.134277 | 8/21 | 1.8250754491656211 | pilot reference |
| 2026-08-25 | pilot p50: 150 epochs, p_max 0.5, ramp 22->90 (jobs 29612707 / 29613556) | pilot | CG 8.315325, RCD 0.878690, RCA 6.639278 | 10/21 | 1.85971801958414 | separation confirmed; gate passed |
| 2026-08-27 | full p25: 575 epochs, ramp 86->345 (train 29613983, sim 29664927) | single | CG 10.310505631407525, RCD 2.104296359202163, RCA 3.3164522148090363 | 10/21 | 1.7207052441497264 | no band upgrade; mean regressed |
| 2026-08-27 | full p50: 575 epochs, ramp 86->345 (train 29613993, sim 29664928) | single | CG 9.190735098536564, RCD 1.1149379652960172, RCA 5.911296139033602 | 10/21 | 1.8112095513767383 | gate 1 pass (RCD 2-5 -> 1-2), gate 2 FAIL (mean > 1.6879978841849728) — [FAIL] |

## 4. Notes

1. Direction re-affirmed from the abandoned branch's note 1: PR #149's
   decomposition makes the training regime the measured lever; #157 and
   #159 both independently name scheduled sampling as their successor
   (#157: the conformity gate is pinned at w~0.08 by teacher forcing;
   #159: the group latent delivered 38% of the required spread move and
   its remainder is the free-running state-tracking loss).
2. The maintainer's pilot requirement replaces the original plan's
   optional teacher-forced preflight (step 10 there) with a stronger
   free-running one: same-protocol sims of short-trained arms measure the
   CG effect directly and price the full runs before they are bought.
3. Steps 1-3 done 2026-08-25. Cherry-pick of aec2fd3+7a34868 applied
   clean; diff vs the abandoned branch empty. Tests: 98 passed on Raven
   (all 8 scheduled-sampling tests), 314 passed / 3 skipped locally.
   Pilot configs committed (8bc675e); only intended keys differ from the
   reference. Caveat noted: the scaled pilot holds full p_max for only 60
   epochs (vs 230 in the 575-epoch design) — enough for separation
   detection, not convergence.
4. Step 4 done 2026-08-25 (commit 766419c). ctrl job 29612646 (2:46),
   p50 job 29612707 (22:35): the unroll costs ~8.2x per epoch (1.506 s vs
   0.184 s), so a full 575-epoch arm prices at ~1.5 h — well inside the
   10 h wall. Ramp verified exact (parquet scheduled_sampling_rate matches
   the analytic schedule). Held-out log_loss: p50 worse than ctrl at every
   eval epoch (+0.128 at e149), gap opening with the ramp — the declared
   teacher-forcing tax; both arms still descending at e149 in 5/5 folds
   (150 epochs is a truncation, not convergence).
5. Incident: a parallel session's rsync --delete sync into the shared
   Raven checkout deleted the pilot configs and reverted train.py twice
   during step 4. Training survived because run.sh materializes job.yml
   under .log/ (sync-excluded) and Python had already imported the right
   module; the p50 arm's scheduled sampling was verified live from its
   log (ss_p lines) and the parquet substitution-rate rows, and ctrl's
   main-train.py run was verified byte-equivalent (scheduled_sampling
   absent -> identical path). Silent failure mode to guard against in
   every later cluster step: old train.py ignores the unknown key, so a
   revert landing before job start would train a control under the
   schedsamp name with exit code 0. Guard: verify from job logs +
   recorded metrics, never from exit codes; re-verify remote src md5 at
   job start for sims.
6. Step 7 gate decision 2026-08-25: PROCEED. (a) Separation confirmed —
   pilot CG 9.5926 (ctrl) vs 8.3153 (p50), spread ratio 0.5955 vs 0.6299
   (human 0.8480, reference 0.5882): ~16% of the reference-to-human gap
   at a truncated dose. (b) Cost: 575-epoch arm ~1.5 h, inside the wall.
   (c) Epochs: both pilot arms still descending at e149; keep E=575
   (matches the reference artifact), re-check on the full run.
   Texture for step 12: schedsamp buys RCD massively (3.4842 -> 0.8787,
   two bands in-pilot) and CG partially, but taxes RCA (3.1343 -> 6.6393)
   and RCB (2.0280 -> 3.3984) at p50, leaving the pilot mean slightly
   worse (1.8251 -> 1.8597) — the §6 anti-correlation pattern. The p25
   arm is the dose hedge; gate 1 can also be satisfied by RCD reaching
   1-2 in the full stack if the mean improves.
7. Step 9 done (jobs 29613983/29613993, 1:14:30 / 1:16:15): ramps exact,
   realized substitution 0.2515 / 0.4995 at e574. Held-out log_loss still
   descending at e574 in both arms (p25 best 1.98229, p50 2.02649, both
   at the final eval) — the pilot's likelihood tax shrank with training
   length; 575 remains a truncation, noted for the write-up. The step-9
   subagent stalled after fetching p25 (session gap); the orchestrator
   fetched p50 and committed. Step 10 sim configs verified
   protocol-identical to the parent except the three intended fields.
8. Step 11 done (sims 29664927/29664928, integrity guard clean, commit
   147029c). Spread ratios: human 0.8480, reference 0.5882, p25 0.5757,
   p50 0.6055 — the pilot's CG separation largely did not survive
   full-length training: against the fully-trained reference the p50 edge
   shrinks to ~7% of the gap, and p25 lands below the reference.
9. Step 12: p50 is the representative arm — it alone satisfies gate 1
   (RCD 2.9419 -> 1.1149) and exhibits the full dose-response. Verdict
   per §2: [FAIL] — gate 2 fails in both arms (p25 1.7207, p50 1.8112 vs
   baseline 1.6880).
10. Mechanism reading: scheduled sampling fixes free-running
    contribution autocorrelation (RCD, strongly dose-responsive
    2.94 -> 2.10 -> 1.11) but collapses the punishment response (RCA
    2.08 -> 3.32 -> 5.91, RCB 1.99 -> 2.21 -> 2.98). Plausible cause:
    substitution decouples the (ground-truth) punishment inputs from the
    substituted own-contribution they actually responded to — round t's
    punishment was dealt to the REAL t-1 contribution, but the model sees
    its own sample, making the punishment-contribution contingency
    inconsistent in training. Any successor must keep the pair coherent:
    substitute punishment-conditioning jointly, restrict substitution to
    weakly-punished cells, or gate the substitution on received
    punishment.
11. CG remains structural: even at p_max 0.5 the free-running ratio
    reaches only 0.605 of the human 0.848 (floor 0.588). Combined with
    #159 (latent alone: 0.662), the two mechanisms attack different
    parts of the deficit and neither suffices alone; latent + a
    punishment-coherent curriculum remains the open combination.
