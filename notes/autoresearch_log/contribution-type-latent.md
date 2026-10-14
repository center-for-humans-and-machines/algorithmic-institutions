# Autoresearch log: contribution-type-latent

## 1. Declaration

- **Slot:** contribution
- **Base model:** reference GNN contributor
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`)
- **Target rows:** **CG** (9.85, band > 5 — primary; band upgrade requires
  < 5). Secondary watch: RCA (1.54) and the C marginals, the known
  anti-correlated collateral.
- **Hypothesis:** Real players have persistent dispositions (free-rider,
  conditional cooperator, altruist). The current model draws every agent from
  the same conditional each round, so simulated trajectories carry no stable
  identity and group means collapse toward the population mean (CG 9.85).
  Giving each agent a per-episode latent type — inferred variationally at
  training time, sampled from the prior at simulation time and held fixed for
  the whole episode — gives trajectories persistent heterogeneity whose
  interaction with the learned peer-response amplifies between-group
  divergence, the mechanism CG measures.
- **Planned change (one change):** add an optional per-agent, per-episode
  latent `z` to `GraphNetwork` (VAE-style: trajectory encoder for the
  posterior q(z|episode), standard-normal prior, z appended to the node
  features at every round; ELBO objective = existing CE + beta * KL),
  enabled only via the contribution training config. Simulation samples
  z ~ p(z) once per agent per episode. No new observed features — z is a
  latent, exactly the lesson of PR #146/#152 ("latent beats conditioning"),
  learned in the architecture instead of bolted on as a copula.
- **Evidence base:** #151 and #154 both name a trajectory-level latent as the
  missing piece; #149 shows the missing correlation is not within-round;
  #153 shows richer peer conditioning alone does not move CG; #157 shows the
  teacher-forced conditional suppresses herding mechanisms — a latent
  bypasses that bottleneck because it is not a conditioning feature.
- **Guards (autoresearch §2):** rows <= 1 must not fall below 11/21; mean
  score must not rise above 1.76.

## 2. Plan

Validated by the orchestrator (targets per §2, all steps legal per §5,
frozen surface untouched).

- [x] 1. `graph.py`: add `AgentLatentEncoder` (GRU over the agent's episode
  trajectory -> mu/logvar), `agent_latent` kwarg on `GraphNetwork`
  (`z_dim = 0` when absent), `z` concatenated onto node features before op1,
  `sample_posterior()` and `sample_prior_z()`.
- [x] 2. `graph.py`: config gating + artifact load-compat — `agent_latent`
  and `z_encoder` in `to_save`; old checkpoints without the keys load with
  `z_dim == 0` and produce identical logits.
- [x] 3. `train.py`: ELBO — CE + beta * clamp(KL, free_bits), beta linear
  warmup over `anneal_epochs`, KL logged; disabled path byte-identical.
- [x] 4. `graph.py` `predict_independent`: sample `z ~ N(0,I)` per node when
  `reset_rnn` (or cache invalid), reuse across rounds; plain attribute cache.
- [x] 5. New training config `group_switching_contribution_50ep_type_latent.yml`
  (dim 4, hidden 20, beta 1.0, free_bits 0.02, anneal 100; else identical to
  reference incl. seed 38381, 575 epochs, flip-doubled 50ep data).
- [x] 6. Raven-only unit test `test_agent_latent.py`: disabled parity,
  posterior shapes/KL, z reaches logits, sim-time z persistence, save/load
  round-trip + legacy load.
- [x] 7. black + flake8 over touched files.
- [x] 8. Train on Raven (`train_cluster.sh ah ...`), fetch artifact + metrics.
- [x] 9. Collapse/leakage gate: per-dim KL above free_bits on >= 2 dims; test
  log-loss not implausibly below baseline. GO / step 10.
- [ ] 10. Contingency (only if 9 fails): collapse -> beta 0.5 / free_bits
  0.05; leakage -> move z concat to the rnn_n input (own-z only).
- [x] 11. Stage-1 sim config `23_2g8a_type_latent_self_gnn_contr_gnn_switch.yml`
  — exact copy of the reference except contribution artifact + output paths.
- [x] 12. Simulate on Raven, fetch `per_round.parquet`.
- [x] 13. Sanity check: between-agent spread in the sim exceeds the reference
  run (z reaches behavior) before evaluating.
- [x] 14. `python -m aimanager evaluate` locally; append results row to §3.
- [x] 15. Decision gate per §2: kept iff CG drops, rows <= 1 >= 11, mean
  <= 1.76; band upgrade iff CG < 5. Stage 2 only on band upgrade; PR either
  way.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-20 | per-agent episode latent (dim 4, ELBO, prior-sampled in sim) | 1 | CG 9.850 -> 9.661 | 10/21 (was 11) | 1.7698 (was 1.7596) | FAIL |

## 4. Notes

1. Hypothesis chosen by the human maintainer from three candidates
   (closed-loop training, recurrent relational core, agent-type latent);
   an `auto/contribution-cg-schedsamp` branch already exists in another
   worktree, so closed-loop training is in flight elsewhere.
2. `encode()` flattens `y_enc` to (B, P, T, L), not (N, T, L) — the
   posterior encoder re-flattens; row-major (batch, player) order makes
   the node axes align.
3. Eval/test log-loss injects the posterior mean, and q conditions on the
   full episode including targets — an informative z lowers test loss by
   construction. The step-9 leakage gate must therefore read test loss
   jointly with the per-dim KL (nats bound the leak), not on its own.
   The shipped artifact is the all-data fold (no test set), so checkpoint
   selection is not affected.
4. All 96 Raven tests pass incl. the 6 new agent-latent tests. Legacy
   caveat found while testing: a latent checkpoint stripped of its keys
   cannot run forward (op1 keeps the widened input width) — real
   pre-latent artifacts are unaffected and covered by a dedicated test.
5. Training submitted 2026-08-20 as SLURM job 29413062 (10 h limit,
   `train_cluster.sh ah .../group_switching_contribution_50ep_type_latent.yml`).
6. Step-9 gate: GO. Training (job 29413062, 11m44s) converged; per-fold
   total KL 1.02-1.83 nats/agent-episode with 1-2 active dims (others at
   the free-bits floor; dim identity rotates across folds). Test
   log-loss 2.51-2.72 vs reference 2.79-2.91; the KL bounds leakage at
   ~0.06 nats/prediction, far below the ~0.25 observed gain, so the
   improvement is predominantly genuine. Step 10 contingency not needed.
7. Stage-1 sim (job 29413359): z reaches behavior (between-agent SD of
   episode means 4.18 -> 4.36; group-spread ratio 0.451 -> 0.470) but CG
   stays 9.66. Verdict FAIL: no band upgrade, and both guards regressed
   (CA 0.77 -> 1.01 pushed rows <= 1 to 10/21; mean 1.760 -> 1.770).
8. Mechanism: independent per-agent types add persistent offsets that
   inflate the group-mean spread AND the individual spread in the same
   proportion -- the ratio stays at the independence floor. The C
   marginals pay for the extra offsets (CA/CB/CD +0.22-0.24). CG needs
   *shared* variance (a per-group latent) or amplifying closed-loop
   dynamics (schedsamp), not individual heterogeneity. Notable
   collateral: PD 2.94 -> 2.42 and RCB 1.93 -> 1.59 improved.
9. No Stage 2 (within-band moves cannot become a success). Ending in a
   [FAIL] PR per section 9.7.
