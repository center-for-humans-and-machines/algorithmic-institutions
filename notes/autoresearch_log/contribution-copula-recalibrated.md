# Contribution copula recalibration: a null result

Status: closed, no model change. Evidence branch `auto/contribution-copula-recalibrated` (PR #198), kept for the record; only this note and the guard script are carried forward.

## The question

PR #197 corrected a serving-path defect: during simulation a timed-out player's contribution was substituted with a default of nine, and the contribution model read that nine through its lagged feature on 2.24% of agent-rounds. Fixing it cost the group-spread row CG on the frontier stack, while the stack whose contributor carries no copula improved.

The hypothesis was that the copula's correlation strength had been calibrated in the presence of the defect and had absorbed it, so a refit would recover the row.

## The answer: refuted, and it could not have been otherwise

The strength is fitted by pairwise-likelihood MLE against **human** histories, teacher-forced, from a training tensor in which a timed-out player's contribution and the lag that follows it are the recorded **0** (`parse_agent_rounds` applies `fillna(0)`). The imputed nine lived only in `environment.update_contribution` on the simulation serving path, which the estimator neither imports nor executes. The calibration never saw the defect.

Refitting against the corrected tree returns:

| | value |
|---|---|
| rho, shipped | 0.03949863621805423 |
| rho, refitted | 0.03949863621805423 |
| delta | exactly 0.0, against a bootstrap CI of width 0.0385 |

All 36 estimate and provenance fields in the two parameter files are identical; only the date and the commit hash differ. The stamped artifact is byte-identical to the shipped contributor. Every one of the 22 evaluation rows moves by exactly 0.0000, and the simulation's per-round output is byte-identical to its parent's.

## What this means

The group-spread row had been **flattered** by the defect, not compensated for. There is no calibration debt, and the post-fix value is the honest one.

Strength is therefore closed as a route to that row: the estimator supports exactly one value, and choosing a larger one because the row scores better is tuning at a metric's own definition, which §5 of the protocol forbids and the calibration script's own pre-flight refuses. Shape is not closed — `copula-missing-state.md` measured the human residual dependence as a round-local shock with a one-round echo rather than the shipped episode-long latent, and with the copula off the state spread sits at 21.36 of the human 27.93 while residual variance is already correct.

## The hazard this caught

The estimator writes the lag-1 ratio under the key `phi` and never writes `phi_final`; the stamping script falls back to the bare `phi`. A refit of the strength alone would have silently stamped `copula_phi = 0.8122` over the frozen 1.0 — a second, undeclared change to a separately frozen parameter, passing every existing assertion.

`scripts/artificial_humans/freeze_phi_in_params.py` carries PR #165's ruling across and refuses to guess. That script is the one artefact of this experiment kept as code.

## Two controls the campaign did not previously have

The **simulation is bit-reproducible** across isolated remote directories, GPU nodes and sessions: both arms reproduced the parent's `per_round.parquet` byte for byte. The **evaluation is bit-reproducible**: re-running it over the parent's own simulation rewrote every score, metric and figure and left the working tree clean.

A before-and-after comparison on unchanged artifacts therefore carries a noise floor of exactly zero. That is what allowed this result to be stated as sharply as it is, and it is the counterpoint to the seed-to-seed floor measured in PR #195, which applies when a model is retrained.
