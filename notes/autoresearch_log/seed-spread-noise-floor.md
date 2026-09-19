# The run-to-run noise floor of the 22-row evaluation

Branch `auto/seed-spread-noise-floor`, based on `origin/auto/punisher-ceiling-fix` at `e230629` (PR #192, the maintainer's current model). A measurement, not an experiment: no model is proposed, no row is declared a target, and no §2 gate applies. The deliverable is an error bar on every row of the evaluation, so that past and future single-seed verdicts can be read against it.

## 1. The question

Every verdict in this project rests on one training run and one simulation, and the protocol has no notion of run-to-run variability. The three most recent experiments were decided on margins that may sit inside it: PR #194's target row finished 8.6% of one noise ceiling short of a band; PR #193's protected-row firing was 1.14 pooled standard errors; PR #194's clearest erosion was 1.96. PR #194 also could not separate whether its new feature shifted the simulated contribution level by a whole point, from 9.32 to 10.36 against the human 9.457, or whether that particular retrain did — its own caveat section says so and asks for two more seeds. Without a noise floor none of these can be read.

The question is therefore not "is model X better" but: **how far does the whole 22-row evaluation move when the only thing that changes is the random seed of a training that everybody already accepts?**

## 2. Setup

**No training.** PR #188 (`origin/auto/copula-seed-ensemble`) already trained five copies of the stimulus-skip contributor with seeds 1–5, the shipped config byte-for-byte, on the full training data. Its log records their cross-validated log losses as 2.0226 / 2.0293 / 2.0221 / 2.0201 / 2.0253 against the shipped artifact's 2.0206, and that the shipped artifact sits as far from the ensemble mean as any member does. The committed metrics parquets on this branch reproduce those numbers (2.0216 / 2.0284 / 2.0221 / 2.0201 / 2.0253). So the shipped contributor is a sixth draw from the same distribution, and the six arms are exchangeable by construction.

**The six arms.** The frontier stack of PR #192, `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling` — the stimulus-skip contributor with its copula, the joint-exodus graph-network switch model, and the ceiling-fixed `lin_multinomial` punisher with its severity copula — with **only** the contribution artifact swapped. The five arm configs differ from the source config in exactly three lines: `contribution_model`, `output_dir`, `figure_name`. The sixth arm is the shipped run itself, the one PRs #193 and #194 used as their baseline, so the spread is anchored on the number those experiments compared against.

**The copula is carried, not recalibrated.** The five members are bare; the frontier contributor is copula-stamped. Each member therefore gets the shipped calibration's `rho = 0.03949863621805423`, `phi_final = 1.0`, `copula_switch_every = 1` copied bit for bit (`carry_contribution_copula_params.py`, taken unchanged from `origin/auto/contributor-ceiling-indicator`, then `make_contribution_copula_artifact.py`). Parameters are frozen per model family when only the marginal changes; recalibrating would have made the arms differ in two things instead of one. Every stamp is verified: all 14 tensors bit-identical to the bare base, and the teacher-forced probabilities on all 7,457 train-split rows unchanged.

**Load-and-differ check** (`scripts/data_analysis/seed_spread_verify_members.py`, Raven login node). All six arms load through `GraphNetwork.load`, carry `y_name = contribution`, have the same 7,061 parameters and the same three copula fields. Pairwise max |delta| over the parameter vector 2.06–3.24, relative L2 ~1.4 — independent draws, not perturbations. The shipped artifact's distance to the members (2.45–3.08) sits inside the members' distance to each other (2.06–3.24), which is PR #188's "sixth draw" finding reproduced on the weights.

**Simulation.** The 23-family protocol untouched: seed 42, 100 episodes, 24 rounds, 2 groups x 8 agents, `save_per_round: true`. The simulation seed and episode count are identical across all six arms, so what varies is the trained model and not the draw. Raven, isolated dir `AI_REMOTE_DIR=~/repros/ai-runs/seed-spread`, one A100 each.

**Evaluation.** All 22 rows, locally, with `PYTHONPATH=<worktree>/src` — the shared venv's editable install resolves `aimanager` to the main checkout, which lacks the RCE row and silently scores 21.

**Analysis.** `scripts/data_analysis/seed_spread_noise_floor.py` writes `plots/data_analysis/evaluation/seed_spread_noise_floor/`.

## 3. Results

(filled in below)

## 4. Verdict

(filled in below)

## 5. Notes

(filled in below)
