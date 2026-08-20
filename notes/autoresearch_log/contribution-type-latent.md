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

(to be filled by the validated step list)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Hypothesis chosen by the human maintainer from three candidates
   (closed-loop training, recurrent relational core, agent-type latent);
   an `auto/contribution-cg-schedsamp` branch already exists in another
   worktree, so closed-loop training is in flight elsewhere.
