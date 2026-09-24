# Merge recommendations

A walkthrough of the open pull requests from the September 2026 work, one decision at a time. The original recommendations (merge the chain, then build on `dev`) are **superseded by the plan below**, agreed on 2026-09-24. Each item keeps its evidence; its first line now states what was decided.

## The adopted plan

Nothing is merged wholesale. `dev` stacks 29 pull requests on top of each other, so taking one fix from it means taking the whole ancestry underneath: failed experiments, abandoned arms, ledger resets. Instead:

1. **`dev` stays a reference, not a base.** It remains the record of the integration and of every branch's verdict (`doc/dev-branch.md` on `dev`); no new work starts from it.
2. **The stack: `autoresearch-optimized-stack`.** The frontier lineage up to PR #179 (#160 → #165 → #171 → #179) plus `main`, with its dead experiment arms removed (non-frontier sim configs and their plot directories, the superseded `herding_copula_v2` and `phihat` artifacts). **PR #181 is not included**: it failed its gate. Every manager experiment from #204 onward ran on #181's contribution model (`…vnode_stimulus_skip_herding_copula`), so those numbers are not expected to reproduce exactly on this stack.
3. **The fixes: issue #224.** One branch off the stack, rebuilt commit by commit from the source commits listed there, then a single pull request:
   - **A. Metric suite:** the RCE row (#199).
   - **B. The punisher reads the round it punishes:** legality rule, test, training configs and the retrained GNN punisher (#184).
   - **C. Timeouts, serve and charge the recorded 0:** both punisher paths (#196), the contribution and switch models (#197), punishment aimed at a timed-out player charged as 0 (#208), the RL manager's view (#205), and `contribution_valid` for the **linear** punisher only, kept if its CV sweep selects it (#196).
   - **D. Wrap-up:** a reference-stack simulation with the new punishers, evaluated on the 22-row suite, plus the documentation.

   Out of scope: the #192 ceiling indicator, the flag for the GNN punisher, the timed-out-player payoff correction (#206, left for the RL work), and everything else on #184 and `dev`.

Where a PR is not mentioned below (the rule, RL and LLM manager work, #202–#223), it is out of scope for this round. Its branch stays as the evidence trail.

## What we learned about the underlying problem

The project builds simulated players that behave like the real people in the experiment, so that a manager can be trained against them. Two questions started this round of work, and both ended somewhere different from where they began.

The first was about the shared randomness in the simulation. Group members decide independently once you know their situation, so any leftover pattern in the model's mistakes means the model is missing part of that situation. It turned out the model already explains almost all of the co-movement between group members: once you account for what it sees, only a small residual is left, and the machinery is tuned to exactly that residual. What it is *not* is a model of anything real. Three separate checks ruled out the obvious explanations. It is not the model being uncertain about itself, because copies trained with different random starts disagree far too little and too briefly. Only about a seventh is information the model could have been given. And the pattern in real people is a one-round echo, not a lasting group trait, whereas the machinery holds its effect for a whole game. So the part of it that does the real work is not a correlation at all. It is a source of variety, propping up something else.

That something else is the second finding, and it is now the one problem worth chasing. Real groups keep drifting further apart as a game goes on: they start out similar and end up spread. The simulated groups start out the same way and then flatten out about two thirds of the way through. The randomness in the models is already the right size, and their moment-to-moment predictions are right; what goes wrong is that the situations the simulation reaches stop spreading. The shared-randomness machinery had been quietly hiding this by manufacturing the missing spread.

Four plausible fixes have now been tested and eliminated: a different output design for the player model, turning up the shared randomness, treating the model's uncertainty about itself properly, and training the model on its own output. One thing did help. Correcting a bug where the simulation showed the models a made-up contribution for players who had timed out moved the measure of variety from 18.9 to 21.4, against 27.9 for real games, and it did so without adding any randomness at all. That is the only genuine progress on this defect so far, and a gap remains. The next step is to measure how strongly a group's drift feeds on itself in real games compared with simulated ones, which can be done from files that already exist.


---

## 1. PR #199 — the RCE response-slope measure

**Decision: adopted as #224 item A1**, rebuilt on the #224 branch rather than merged directly.

This is what takes the evaluation suite from 21 rows to 22, and it sits underneath everything else. It was originally bundled inside PR #184 and has been split out so it is reviewable on its own; #184 contains the same commit and will see it as already present.

RCE fits, per contribution band, the slope of the next-round contribution change on the punishment received. It measures the causal channel a trained manager would exploit, which the existing RCB row does not: RCB bins by punishment rate, mixing contribution level with dose, and its bin averages can be matched by models whose players respond in the **wrong direction**.

| | RCB | RCE |
|---|---|---|
| correlation with the count of human-signed band slopes | -0.08 | -0.88 |
| rank correlation between the two rows, over 40 stacks | 0.28 | |

RCE's noise ceiling is large relative to the effect, so it is added **beside** RCB rather than replacing it.

It also matters for a practical reason: until it lands, anyone checking out `main` gets the old 21-row suite, which is the trap that repeatedly caused the shared environment to silently drop the protected measure.

---

## 2. PR #184 — the manager reacts to the round it can actually see

**Decision: adopted in part, #224 items B1–B5.** Taken: the legality rule and its documentation (`7ad1ddd`), the test (`ce70a09`), the two training configs (`c74bc0e`, configs only) and the retrained GNN punisher (`44997a6`). The linear punisher is retrained on the #224 branch (C7) instead of taking `224c86e`'s artifact. Not taken: the ledger reset, the re-run simulations, the `_curpun` configs and the copula ancestry.

**How the simulation times punishment, confirmed on `main`.** In `simulate.py`'s round loop, round t's contributions are drawn, then `get_punishments(rounds + [round_dict])` is called, then `env.punish` and `env.step`. Punishment comes right after that round's contributions, and both punisher input paths already hold c_t in the last slot of `contribution`. Only the feature-legality rule kept the punishers from reading it.

The simulated manager had been deciding punishment from the *previous* round's contribution. Real managers punish what you just gave. Three independent checks confirmed it: the payoff accounting in the data only balances with same-round punishment, punishment correlates with the current contribution more strongly than the previous one, and the decisive contrast is stark. Real managers punished someone who had just dropped from 20 to 4 more than half the time, and someone who had just risen from 4 to 20 less than a fifth of the time. Every simulation had that backwards or flat.

This was true on every branch for the project's whole history, and a validation rule actively blocked fixing it, so the automated experiment loop could never have found it.

Fixing it moved the manager-policy row from the 1.2 to 1.6 range into 0.7 to 0.9 in all six stacks tested, and the binned punishment-reaction row down by 0.5 to 0.9 everywhere.

This pull request also carries the new response-slope measure and the reset of the ledger's baselines, since the fix moves every row of every stack.

It is a correctness fix with independent verification, and nothing else in the chain means anything without it.

---

## 3. PR #192 — the manager can tell a maximum contribution apart

**Decision: not taken in this round.** A verified mechanism fix, but independent of the timing and timeout fixes; it can be extracted the same way later.

Real managers almost never punish someone who contributed everything, but hit hard on the rare occasions they do. The simulated manager punished them constantly and softly. The cause was that contribution entered as a single number, so the model could not express a sharp break at exactly 20 and interpolated the ceiling from the band below.

Adding one indicator fixed it almost exactly.

| | real | before | after |
|---|---|---|---|
| chance a full contributor is punished | 3.9% | 12.2% | 4.0% |
| how hard, when it happens | 7.0 | 3.8 | 8.0 |

It failed its formal gate, because the row it targeted improved substantially without crossing a band boundary. The gate exists to stop score-chasing, and this is a correctness fix that happens not to cross a line.

One thing it did not fix, which matters later. The reaction-at-the-ceiling measure still did not move, and the reason turned out to be on the players' side rather than the manager's: they under-react to heavy punishment at the ceiling by about 2.3 times.

---

## 4. PR #196 — the manager can tell a timeout apart from a genuine zero

**Decision: taken in part, #224 items C1, C2, C6 and C7.** Serving the recorded 0 on both punisher paths is taken (`838924d`, ported by hand), with its tests (`256ae99`). `contribution_valid` enters the linear feature pool (`f0ed57e`) and the linear punisher's CV sweep decides whether it is used. The GNN punisher does not get the flag, and none of this branch's artifacts are taken.

When a player times out, the game records their contribution as zero. So the manager saw two kinds of zero: people who chose to give nothing, and people who never answered. Real managers punished the first kind 43.5 percent of the time and the second kind never. The simulated manager could not tell them apart.

Giving it a flag for whether the player answered helps, but only on one of the two manager models.

| | fit, before | fit, after |
|---|---|---|
| linear manager | 1.3446 | 1.3271 |
| graph manager | 1.1743 | 1.1719 |

The linear one also moved its sensitivity to contribution from -0.125 to -0.173 against the human -0.242, closing 41 percent of that gap. The graph one gained nothing statistically, moved its sensitivity the *wrong* way, and its overall fit got slightly worse. On a stack using the graph manager, the evaluation degraded clearly.

The gain is confined to one of the two model families and does not justify carrying the change. The finding is kept as a record; the code is not merged.

---

## 5. PR #197 — the players see what actually happened

**Decision: adopted as #224 item C3** (`26386bc`, ported by hand onto the stack).

The same timeout problem as #196, but one layer down. During simulation, a timed-out player's contribution was being replaced with a default of nine before the models were shown it, on 2.24 percent of agent-rounds. The contribution model was reading that nine through its memory of the previous round, on every affected cell. After the fix it reads the recorded zero.

The measurement that matters:

| | variety of situations the model reaches |
|---|---|
| real games | 27.9 |
| before | 18.9 |
| after | 21.4 |

That is direct progress on the one defect the campaign has left, and it was gained without adding any randomness. Nothing was retrained, so the comparison carries none of the training noise, which makes it unusually clean.

It failed its gates because group-spread rows worsened on the frontier stack. PR #198 proves that cost is honest: the bug had been inflating that row. The other stack tested, whose player model carries no shared-randomness machinery, actually improved.

This is the best result of the September 2026 work.

---

## 6. PR #198 — the copula recalibration null result

**Decision: close without merging.** Already actioned: condensed onto `main` as commit `c67308b`.

This tested whether the shared-randomness setting had been calibrated against the bug PR #197 fixed. If it had, the setting would need refitting and the group-spread loss would be recoverable.

It had not, and could not have been. That setting is fitted against the human data, where the value was always recorded correctly. The bug lived only in the simulation path, which the fitting procedure never touches. Refitting returned the identical number to the last digit, the resulting model file is byte-identical to the one already in use, and every one of the 22 measures moved by exactly zero.

So the group-spread row had been flattered by the bug, not compensated for. There is no debt to repay, and that route to recovering the row is closed: the estimator supports one value, and picking a larger one because it scores better would be tuning against the measure's own definition.

Because nothing changed, the pull request's 62 files are evidence for a null result — figures and tables identical to ones already in the repository, plus a duplicate model file. Rather than merge that, two things were carried onto `main` directly:

- `notes/autoresearch_log/contribution-copula-recalibrated.md`, describing the experiment and result.
- `scripts/artificial_humans/freeze_phi_in_params.py`, the guard. The estimator writes the lag-1 ratio under the same key the stamping script reads for persistence, so refitting the strength alone would have silently replaced the frozen persistence of 1.0 with 0.81, passing every existing check. A prose description of a guard protects nothing, so this stays as code.

The branch is kept for the evidence trail; the pull request can be closed.

**Note:** this condensing does not generalise to the other result-only pull requests. Their tables are cited directly by `notes/autoresearch.md` and by both HTML reports, so those have to remain as files.

---

## 7. PR #189 — the rules

**Decision: not decided; not part of this extraction.**

This is the only pull request that changes how the project judges work rather than what the models do. Four things in it.

**The shared-randomness settings are frozen**, so a change to a model and a change to the noise can no longer move at the same time and be mistaken for each other. That freeze is what caught the hazard recorded in item 6.

**Player-model changes are judged with that machinery switched off**, against a direct measure of how much variety the simulation retains, rather than through a score the machinery dominates.

**The gates are noise-aware, and symmetric.** A target must beat its measurement noise to count as an improvement, and equally, a row worsening by less than its noise is not a cost. Applying the threshold only to gains would have made improvement nearly impossible: one row would have to beat the noise to help, while twenty-one could hurt by luck. The 22-row mean, whose margin is about two noise widths, remains the backstop that catches many small real losses accumulating.

**The frontier's baseline is an average of six equivalent models** rather than one lucky run.

It also carries the plan file recording what was attempted and how it turned out, including two hypotheses that measurement killed.

Without it none of this binds anyone, and the three experiments already judged under the new rules reference it.

---

## 8. PRs #183, #186, #187, #188, #191, #195 — the six measurements

**Decision: not taken in this round.** The consolidation is PR #200 (73 files against the original 759); the six branches stay as the evidence trail.

None changes a model's behaviour. Between them they established: the punishment response is learned rather than memorised; the shared-noise machinery refills collapsed variety rather than restoring a correlation; observable missing information explains about a seventh of the residual correlation and the static component is zero; model uncertainty is about a sixth of the noise dose and has the wrong timescale; a location-scale output head extrapolates worse, not better; and retraining the identical model moves a typical row by 0.138, with ten of 22 rows unjudgeable on one run.

Merging all six intact would be 759 files as GitHub counts them, 478 counting only each branch's own commits. Consolidated to 73: the six logs, the tooling scripts (several already reused downstream), two behaviour-neutral source additions, the reusable trained models, and every table cited elsewhere.

The rule applied: keep the inputs and the script that regenerates a derived artifact, drop the derived artifact. The one exception is the five seed-trained models, which are derived from nothing and cost ten GPU-minutes each.

Both atlas generators read their numbers straight out of these six branches, so the consolidation repointed them at `main` and rebuilt both reports byte-identically, verified twice with the cache cleared. It found 29 such paths, including one pointing at a temporary agent worktree that will not outlive these pull requests, and caught a cache collision that would have silently served the wrong numbers into a report.

Two judgement calls: the ten fold-trained models were dropped, since nothing downstream re-measures from them and they are inert on `main`; and of the two source additions, the ensemble sampler was carried but the hidden-variable logging was not, because alone it would add a permanently empty column to every simulation output.

**Conflict to expect:** this and PR #185 both touch the two generator files. Whichever merges second conflicts there; the resolution is small and documented in #200's body.

Everything in it is inert on `main` today: the tooling imports modules that exist only on the model lineage. The measurements are the deliverable; the scripts are carried to be runnable later. The branches stay as the evidence trail.

---

## 9. PRs #190, #193, #194 — the three failed attempts

**Decision: close without merging.**

- **#190** ported the group-switching component from the other model family. Composition measures improved, response measures worsened. One declared target turned out not to be a switching measurement at all — a specification error, not a model failure.
- **#193** gave the manager a richer contribution encoding. Refuted on human data before any compute was spent. Its decomposition is the durable part: two thirds of the gap it targeted was an artefact of the training split, not a model limitation.
- **#194** gave the players a maximum-contribution indicator. Mechanism installed and the target nearly crossed; failed on the protected measure. PR #195 later showed four of the five side effects it was faulted for are reproduced by retraining the same model.

The project's convention is to leave failures open as the ledger. These are closed instead, with their conclusions folded into the consolidated note so the knowledge sits in one place and the branches remain as evidence.

One thing worth carrying into code: #193's screening script, which refutes a contribution-model hypothesis on human data in seconds before a training run is spent.

---

## 10. PR #185 — the two illustrated reports

**Decision: not taken in this round** (it depends on item 8).

Two self-contained HTML reports and their generators. One is written for readers with no prior context; the other reproduces the format of the existing campaign atlas. Both are published, both are current as of this work, and both are regenerable from committed data.

Order matters: the generators read their numbers from the six experiment branches, and item 8 repoints them at `main`. Merging this first would leave the reports depending on branches that are about to be closed.

---

## 11. Later pull requests carried by #224

- **PR #208** — a punishment aimed at a player who gave no input is charged as 0, as the game charged it. **#224 item C4** (`bb73607`).
- **PR #205** — the RL manager is served the recorded 0 for a timed-out player. **#224 item C5** (`d83cfee`).
- **PR #206** — the timed-out player's payoff correction shares a commit with the `common_pool` RL reward mode. **Not taken**; it belongs with the RL work.
