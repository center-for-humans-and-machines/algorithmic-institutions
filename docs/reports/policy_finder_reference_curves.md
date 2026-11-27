# Reference curves of the Opus policy-finder campaigns, re-evaluated on DAIS

The success criterion of `docs/plans/policy-finder-open-weights.md`: an open-weight run reaches at least 70% of the Opus run's improvement, where the improvement is the final best rule's score minus the best first-generation rule's score, both scored with the same simulator on DAIS. This report gives that improvement and its 70% bar for the two Opus reference campaigns, at a precision that can carry the comparison: experiment 2, `pf-20261104` (`docs/reports/policy_finder_proxy_vs_cem.md`), and experiment 1, `pf-20261105` (`docs/reports/policy_finder_exp1_vs_rl.md`). Measured on 2026-11-27, DAIS jobs 988331 (exp2) and 988332 (exp1), 4.5 minutes each on 92 CPUs of `small`; the earlier runs 988119 and 988120 gave identical per-version results.

## Summary

| | exp2 `pf-20261104` | exp1 `pf-20261105` |
|---|---|---|
| Best first-generation rule (by recorded score) | `-7`: 69.84 ± 0.29 | `-3`: 649.61 ± 0.20 |
| Final best rule (by recorded score) | `-12-v7`: 73.39 ± 0.29 | `-7-v3`: 655.37 ± 0.19 |
| **Improvement** (paired; 95% interval) | **3.55 ± 0.09** (3.36 to 3.74) | **5.76 ± 0.09** (5.58 to 5.94) |
| **70% bar** (95% interval) | **2.49 ± 0.06** (2.35 to 2.62) | **4.03 ± 0.06** (3.91 to 4.16) |
| Had the runner-up of generation 1 been selected | 3.36 ± 0.11 (bar 2.35) | 6.07 ± 0.10 (bar 4.25) |
| Selection-robust improvement (cross-fit) | 3.36 ± 0.12 (3.07 to 3.64) | 5.77 ± 0.09 (5.59 to 5.95) |
| Episodes per version | 30,000 (15 slices of 2,000) | 90,000 (45 slices of 2,000) |
| Versions | 36: 35 with a final run, `-7-v7` failed | 36, all with a final run |

The scores are re-evaluated on fresh seeds; "recorded" is the score the campaign gave the rule (its 2,000-episode final run, seed 1000) and is what the campaign selected on. From single 2,000-episode final runs the same improvement read 3.1 and 5.0, with errors too large to judge 70% of it; here its standard error is about 0.09, but that error covers only the measurement of the two rules the campaign happened to select, not the selection itself, which was close to a coin flip in generation 1 of both campaigns (next section). The intervals use the t value for the number of slices (2.145 for 15, 2.015 for 45).

## Method

`infra/dais/reevaluate.py` takes a campaign's membership (campaign.py's state file, `{"generations": [{"index", "members": [{"name", "version"}]}]}`) and, for each version, the files on its branch: the final run's config and its scored `sweep.json`, whose `best_name` is the GP peak the campaign scored, and the rule and code of the commit that added that config. A version without them is listed as "no final" and counts as no improvement, as in campaign.py.

- **What is played:** the version's GP peak alone, with its params from the final config, against the same managers as in its final run (`ah` in exp2; alone in exp1).
- **Seeds:** K slices of the final run's size (2,000 episodes, two batches of 1,000), slice j on seed 100,000 + 2j. Every version plays the same slices (common random numbers), and no seed the campaigns used.
- **Episodes:** K is the smallest number with (largest recorded GP-peak standard error of the campaign) / sqrt(K) at most 0.3: 1.15 gives 15 for exp2, 2.00 gives 45 for exp1 (exp1's largest is an outlier; its typical recorded error is 1.3).
- **Errors:** a score's error combines its slices' own. A difference between versions (each generation's gain over generation 1's best, and the improvement) is paired slice by slice, with its error from the spread of the 15 or 45 per-slice differences. Because the slices share their games across versions, the paired error is about a quarter of what independent errors would give (0.09 against 0.41 for exp2).
- **Selection:** each generation's best so far is chosen by recorded score, as the campaign chose, and reported with its re-evaluated score. Re-evaluating removes the winner's bias from the reported scores, but which rule was selected still depends on the recorded scores' noise (next section).
- **Shape:** single-threaded sims in the local backend's pool (`PF_CLUSTER_BACKEND=local`), the sim image unchanged, each version on a snapshot of its own commit.

The campaigns' membership is reconstructed in `docs/reports/policy_finder_reference_curves/<prefix>.campaign.json`: generation 1 is v1 of the 12 lineages, and the same four lineages held every later generation's slots (exp2: `-12`, `-9`, `-5`, `-7`; exp1: `-3`, `-7`, `-8`, `-11`; as both reports state), so generation g holds their version g. The branches also hold `pf-20261104-1-v2` and `-11-v2`, which stopped before their sweep and are not part of the campaign.

## Selection noise

The improvement is measured between two rules the campaign selected on its recorded scores, which carry an error of about 1.1 (exp2) and 1.3 (exp1). In generation 1 of both campaigns the best and the runner-up were tied within that error, so which one defines the start of the improvement was close to a coin flip:

| | exp2 | exp1 |
|---|---|---|
| Generation 1's best by recorded score | `-7`: 70.50 | `-3`: 649.75 |
| Its runner-up | `-12`: 70.33 | `-7`: 649.70 |
| Improvement from the best (the criterion's) | 3.55 ± 0.09 | 5.76 ± 0.09 |
| Improvement from the runner-up | 3.36 ± 0.11 | 6.07 ± 0.10 |
| The 70% bar moves by | 0.13 | 0.22 |

So the bar moves by 0.13 (exp2) and 0.22 (exp1) on selection noise alone, two to four times its quoted error of 0.06, and an open-model run's improvement carries the same kind of noise. The final best is a close call too. In exp2, `-12-v7` (recorded 73.43) led `-12-v5` (73.29), and it also leads when re-evaluated (73.39 against 73.12), so the other choice would have lowered the improvement by 0.27. In exp1, `-7-v3` (656.05) led `-3-v4` (655.91), and re-evaluated they are 655.37 and 655.27, so the other choice would have changed the improvement by only 0.10.

The tool therefore also reports, for any campaign, improvements that do not hinge on the campaign's selection (`summary.json`, `robust`):

- **in sample:** from generation 1's best to the best overall, both chosen by re-evaluated score on all slices. Optimistic, because the choice and the estimate share their noise: exp2 3.36 ± 0.11 (`-12` to `-12-v7`), exp1 5.76 ± 0.09 (`-3` to `-7-v3`).
- **cross-fit:** the same choice made on one half of the slices (even or odd) and measured on the other half, both ways, averaged. No shared noise between choice and estimate: exp2 **3.36 ± 0.12** (95% interval 3.07 to 3.64), exp1 **5.77 ± 0.09** (5.59 to 5.95).

For a comparison between campaigns, the cross-fit improvement of the open-model campaign against the cross-fit improvement here (70%: 2.35 for exp2, 4.04 for exp1) does not depend on a coin flip in either campaign. The criterion as written (from the campaign's own selection) gives the bars of the summary table; both should be reported.

## Simulator code

The tool plays each version with the code of its own commit, while an open-model campaign will run with the integration branch's code: `src/aimanager` at `origin/policy-finder-open-weights` (97c0b8af) differs from the reference campaigns' trees (`rule.py`, `sweep.py`, `simulate.py`, `pool_scores.py`). To check whether this matters, the four rules that define the bars (exp2 `-7`, `-12-v7`; exp1 `-3`, `-7-v3`) were re-evaluated with each version's rule and configs but the current `src/aimanager` (`reevaluate.py --code origin/policy-finder-open-weights`, DAIS jobs 988339 and 988340), on the same 15 and 45 slices and seeds.

All 120 per-slice results (30 for exp2, 90 for exp1) are identical to 6 decimals to those under each version's own code.

That identity is real, not a sign that both runs used the same code. Run in the production container setup (`local_cluster.run`, the same Apptainer command as every sim) on a reference snapshot (`pf-20261104-12-v7`'s final commit, fa0920c6) and on the current one (02c0a475), Python imports `aimanager` from `/opt/algorithmic-institutions/src/`, the mounted snapshot, and the imported files' SHA-256 match each commit's git blobs: `simulation/simulate.py` 340d5665… (reference) against 50f8e4d2… (current), `manager/rule.py` da0e1576… against 40c8c0df…. The image has no `aimanager` of its own (`import aimanager` fails in it without the snapshot); its own environment only puts `/opt/algorithmic-institutions/src` on the path. Since then every sim on the local backend checks this itself before it runs: if the imported `aimanager` does not lie in the snapshot's `src/`, the sim fails with exit code 86. The code changes are behaviour-neutral for these rules and settings, so a new campaign scored with current code can be compared with these bars directly. If later changes to `src/aimanager` touch the simulation, the same check (`--code <ref>`, about 40 s per campaign pair) shows whether the references must be re-evaluated.

## Curves

Best so far by recorded score, its re-evaluated score, and its paired gain over generation 1's best.

exp2 `pf-20261104`:

| gen | best so far | recorded | re-evaluated | gain over gen 1 |
|---|---|---|---|---|
| 1 | `-7` | 70.50 ± 1.12 | 69.84 ± 0.29 | 0 |
| 2 | `-9-v2` | 71.91 ± 1.12 | 71.92 ± 0.29 | 2.08 ± 0.17 |
| 3 | `-12-v3` | 73.02 ± 1.13 | 72.84 ± 0.29 | 3.01 ± 0.08 |
| 4 | `-12-v3` | 73.02 ± 1.13 | 72.84 ± 0.29 | 3.01 ± 0.08 |
| 5 | `-12-v5` | 73.29 ± 1.12 | 73.12 ± 0.29 | 3.29 ± 0.10 |
| 6 | `-12-v5` | 73.29 ± 1.12 | 73.12 ± 0.29 | 3.29 ± 0.10 |
| 7 | `-12-v7` | 73.43 ± 1.12 | 73.39 ± 0.29 | 3.55 ± 0.09 |

exp1 `pf-20261105`:

| gen | best so far | recorded | re-evaluated | gain over gen 1 |
|---|---|---|---|---|
| 1 | `-3` | 649.75 ± 1.30 | 649.61 ± 0.20 | 0 |
| 2 | `-3-v2` | 651.87 ± 1.29 | 651.71 ± 0.20 | 2.10 ± 0.09 |
| 3 to 7 | `-7-v3` | 656.05 ± 1.25 | 655.37 ± 0.19 | 5.76 ± 0.09 |

- Every step of both curves by recorded score is also a step up when re-evaluated: the campaigns' selections hold.
- Recorded scores lie above the re-evaluated ones by 0.23 on average over the 35 exp2 versions (sd 0.42) and by 0.50 over the 36 exp1 versions (sd 0.48). All recorded finals share seed 1000 and all re-evaluations seeds 100,000 and up, so most of this shift is a difference between the two sets of games, not between rules.
- The best first-generation rule by re-evaluated score is a different one in exp2 (`-12`, 70.03 ± 0.29, against `-7`'s 69.84): the two are tied within their errors. The criterion uses the campaign's own selection (`-7`), so the improvement is measured from it.

## Data

In `docs/reports/policy_finder_reference_curves/`, per campaign prefix:

- `<prefix>.campaign.json`: the membership given to the tool.
- `<prefix>.versions.tsv`: generation, lineage, version, stem, status (`scored` or `no final`), recorded score and se, re-evaluated score and se, episodes and the commit played.
- `<prefix>.curve.tsv`: the curves above.
- `<prefix>.summary.json`: the improvement, its se and 95% interval, the 70% bar, the selection-robust companions (`robust`), the number of slices and the versions without a final.
- `<prefix>.slices.tsv`: every version's score in every slice, the raw numbers behind the paired errors.

The tables are TSV because the repo ignores `*.csv` (`docs/large_files.md`).

## Putting an open-model campaign through the same tool

On DAIS, from a checkout with the campaign's instance branches (its `PF_RUNS`):

```bash
sbatch infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    $PF_RUNS/campaign/<campaign>.json <out dir>
```

The defaults (`--seed 100000`, `--target-se 0.3`) give the same seeds as here, so the open model's rules play the reference rules' games. Its improvement is then compared with the bar above; the bar's own error (0.06) is small next to the open run's.

Not covered: a campaign whose score is a gain over a baseline (exp2_diff) is ranked by its recorded pool here, not by its gain; that setting is not among the references.
