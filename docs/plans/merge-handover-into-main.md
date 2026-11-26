# [DRAFT] Merge `yannik/handover` into `main`

## State (2026-11-26)

- `main` is at `69451452` (2026-10-01, "Draft a plan for an LLM-backed manager"), and every
  commit on it is in `yannik/handover`. The merge is a fast-forward: no merge commit, no
  conflicts.
- `yannik/handover` is at `4fad2fee`. That is 278 commits on top of `main`, dated 2026-08-25 to
  2026-11-13. Their authors are Cem (136), Claude Code / Claude (122), Yannik (11) and
  Levin (9).
- Diff: 1413 files, +64k / −1.4k lines.
  - 1087 files added, 235 deleted, 45 modified, 46 renamed.
  - 667 of the added files are JPEGs under `plots/simulation` (35 MB).
  - Two added files are over 1 MB: `docs/reports/zero_punisher_test.html` (3.2 MB) and
    `experiment_analysis/algoinst_profit_real_experiment.ipynb` (1.9 MB).
- No credentials turned up in the added lines; the only hits were base64 images.

## Issues to decide before merging

1. **Git LFS goes away on `main`.**
   - Handover removed LFS on 2026-11-03 (`1a0d5dcf`). `*.csv`, `*.parquet` and `*.pt` are
     gitignored, and `.gitattributes` has no LFS filters.
   - `main` tracks 160 such files (the human datasets in `experiments/`, the clones' `.pt`, sim
     outputs). After the merge none of them are tracked, and `git pull` deletes them from
     everyone's checkout.
   - They come back with `scripts/fetch_large_files.py` (`docs/large_files.md`). That needs
     read access to `s3://chm-algorithmic-institutions`, or git-lfs plus `--lfs` for the 177
     files that are only in the LFS history.
   - Everyone working off `main` has to know this beforehand.
2. **Branches that still use LFS will conflict later.**
   - `dev` (Levin, 177 commits not in handover) changes 483 data files relative to `main`.
     Merging `dev` into `main` afterwards gives modify/delete conflicts on deleted files.
     Data files it adds would come in as plain LFS pointer text, because handover's
     `.gitattributes` no longer has the filter.
   - The same applies to the older `auto/*`, `llm-manager-*` and `base/*` branches.
   - Levin's active `annot/step-classification` branched off handover (2026-11-06) and touches
     no data files, so it is unaffected.
3. **`dev` is not part of this merge.**
   - `main` and `dev` already diverge: 5 commits only on `main`, 234 only on `dev`.
   - After the merge, `main` carries the handover line, and `dev`'s RL / LLM-manager work stays
     outside it.
   - Decide with Levin and Cem which branch is the trunk. If it is `dev`, this merge goes
     there instead, or both.
4. **Layout moves.** `doc/` and `reports/` move to `docs/`. Links, bookmarks and scripts that use
   the old paths break.
5. **Deletions.**
   - Cem's PR #235 removes the rule-based managers' rules, sims and reports.
   - Yannik's "takeover" commit `77803906` deletes 45 files, among them `reports/*` and
     `reports/zero_punisher_test.html`.
   - Older artifacts under `artifacts/artificial_humans/*` and `artifacts/manager/*` are gone
     as well.
   - All of these are deliberate on handover, but they are new to anyone on `main`.
6. **Experimental code on handover.** The common-games policy-finder changes (`4fad2fee` and
   the two commits before it) were merged without their validating campaign. See
   `docs/plans/policy-finder-common-games.md`, still ACTIVE. They only affect the `exp2_diff`
   setting.
7. **Tests.** Two tests fail on handover, as they did before the merge:
   `test_punishment_copula::test_real_bundle_group_spread_beats_independent` and
   `test_group_vnode::test_flip_augmentation...`. Fix them or mark them xfail first, or accept
   them.
8. **Repository size.** About 40 MB of rendered plots and reports become part of `main`'s
   history: the JPEGs, the HTML report and the notebook. They can stay, or be dropped from the
   merge, which needs an extra commit.
9. **Process.** `gh` is not installed here, so branch protection on `main` is unchecked. If
   `main` requires a PR, the merge goes through a PR from `yannik/handover`, not a direct push.

## Steps

1. Agree on 1–3 with Levin and Cem: the LFS removal, the trunk, and who fetches data how.
2. Optional cleanup on handover, then push:
   - fix or xfail the two tests;
   - decide on the large rendered files;
   - archive or close the common-games plan.
3. Tag the current `main` (`pre-handover-main`) so the LFS-era tree stays easy to find.
4. Merge:
   - `git checkout main && git merge --ff-only yannik/handover && git push origin main`,
   - or open a PR `yannik/handover` → `main` and merge it as a fast-forward / merge commit,
     without squashing, so the instance branches' history stays reachable.
5. Announce:
   - the LFS removal and `scripts/fetch_large_files.py`;
   - the `docs/` move;
   - for `dev` and the other LFS branches, rebase or merge onto the new `main`, dropping
     their LFS pointers.
6. Keep working from `yannik/handover`, or move the workflow to `main`. This decides the base
   in the branch workflow.
