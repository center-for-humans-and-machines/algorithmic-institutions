# Reference curves of the Opus policy-finder campaigns, re-evaluated on DAIS

The success criterion of `docs/plans/policy-finder-open-weights.md`: an open-weight run reaches at least 70% of the Opus run's improvement, where the improvement is the final best rule's score minus the best first-generation rule's score, both scored with the same simulator on DAIS. This report gives that improvement and its 70% bar for the two Opus reference campaigns, at a precision that can carry the comparison: experiment 2, `pf-20261104` (`docs/reports/policy_finder_proxy_vs_cem.md`), and experiment 1, `pf-20261105` (`docs/reports/policy_finder_exp1_vs_rl.md`). Measured on 2026-11-27, DAIS jobs 988119 (exp2) and 988120 (exp1), 4.5 minutes each on 92 CPUs of `small`.

## Summary

| | exp2 `pf-20261104` | exp1 `pf-20261105` |
|---|---|---|
| Best first-generation rule (by recorded score) | `-7`: 69.84 ± 0.29 | `-3`: 649.61 ± 0.20 |
| Final best rule (by recorded score) | `-12-v7`: 73.39 ± 0.29 | `-7-v3`: 655.37 ± 0.19 |
| **Improvement** (paired) | **3.55 ± 0.09** | **5.76 ± 0.09** |
| **70% bar** | **2.49 ± 0.06** | **4.03 ± 0.06** |
| Episodes per version | 30,000 (15 slices of 2,000) | 90,000 (45 slices of 2,000) |
| Versions | 36: 35 with a final run, `-7-v7` failed | 36, all with a final run |

The scores are re-evaluated on fresh seeds; "recorded" is the score the campaign gave the rule (its 2,000-episode final run, seed 1000) and is what the campaign selected on. From single 2,000-episode final runs the same improvement read 3.1 and 5.0, with errors too large to judge 70% of it; here its standard error is about 0.09, so an open-model run's improvement can be set against the bar with an error dominated by its own runs.

## Method

`infra/dais/reevaluate.py` takes a campaign's membership (campaign.py's state file, `{"generations": [{"index", "members": [{"name", "version"}]}]}`) and, for each version, the files on its branch: the final run's config and its scored `sweep.json`, whose `best_name` is the GP peak the campaign scored, and the rule and code of the commit that added that config. A version without them is listed as "no final" and counts as no improvement, as in campaign.py.

- **What is played:** the version's GP peak alone, with its params from the final config, against the same managers as in its final run (`ah` in exp2; alone in exp1).
- **Seeds:** K slices of the final run's size (2,000 episodes, two batches of 1,000), slice j on seed 100,000 + 2j. Every version plays the same slices (common random numbers), and no seed the campaigns used.
- **Episodes:** K is the smallest number with (largest recorded GP-peak standard error of the campaign) / sqrt(K) at most 0.3: 1.15 gives 15 for exp2, 2.00 gives 45 for exp1 (exp1's largest is an outlier; its typical recorded error is 1.3).
- **Errors:** a score's error combines its slices' own. A difference between versions (each generation's gain over generation 1's best, and the improvement) is paired slice by slice, with its error from the spread of the 15 or 45 per-slice differences. Because the slices share their games across versions, the paired error is about a quarter of what independent errors would give (0.09 against 0.41 for exp2).
- **Selection:** each generation's best so far is chosen by recorded score, as the campaign chose, and reported with its re-evaluated score. Selecting on noisy recorded scores and then re-evaluating removes the winner's bias from the reported numbers.
- **Shape:** single-threaded sims in the local backend's pool (`PF_CLUSTER_BACKEND=local`), the sim image unchanged, each version on a snapshot of its own commit.

The campaigns' membership is reconstructed in `docs/reports/policy_finder_reference_curves/<prefix>.campaign.json`: generation 1 is v1 of the 12 lineages, and the same four lineages held every later generation's slots (exp2: `-12`, `-9`, `-5`, `-7`; exp1: `-3`, `-7`, `-8`, `-11`; as both reports state), so generation g holds their version g. The branches also hold `pf-20261104-1-v2` and `-11-v2`, which stopped before their sweep and are not part of the campaign.

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
- `<prefix>.summary.json`: the improvement, its se, the 70% bar, the number of slices and the versions without a final.

The tables are TSV because the repo ignores `*.csv` (`docs/large_files.md`).

## Putting an open-model campaign through the same tool

On DAIS, from a checkout with the campaign's instance branches (its `PF_RUNS`):

```bash
sbatch infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    $PF_RUNS/campaign/<campaign>.json <out dir>
```

The defaults (`--seed 100000`, `--target-se 0.3`) give the same seeds as here, so the open model's rules play the reference rules' games. Its improvement is then compared with the bar above; the bar's own error (0.06) is small next to the open run's.

Not covered: a campaign whose score is a gain over a baseline (exp2_diff) is ranked by its recorded pool here, not by its gain; that setting is not among the references.
