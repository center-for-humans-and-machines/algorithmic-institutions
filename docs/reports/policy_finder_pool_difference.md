# Policy finder scored by the lead over `ah`

Plan: `docs/plans/archive/policy-finder-pool-difference.md`.
Campaign: `exp2-diff-2026-11` (`pf-20261112-*`).
Head-to-head: `34_DIFF_h2h`, seeds 20000–31000.

## Summary

- **The campaign's rules beat never punishing.** On fresh seeds the three lineage bests lead `ah`'s
  group by 15.65–15.96. Never punishing (`zero`) leads by 14.23. The gain is +1.42 to +1.73,
  ± 0.42, so 3–4 standard errors.
- **They do not beat an earlier rule that was never optimised for the lead.**
  - That rule is the multi-opponent campaign's `pf-20261106-9-v5` with its opponent weight set to
    0. It leads `ah` by 17.11 ± 0.33.
  - The best new rule, `-6-v6`, is 1.15 ± 0.44 behind it.
  - The earlier pool-maximisers sit between `zero` and the new rules: `prev_best` 14.89, Cem's best
    15.08.
- **The new rules almost never punish.** They fine only low givers and leave everyone else alone.
  `-6-v6` and `-5-v7` fine hard once or twice, right after the first switch decision. `-2-v7`
  fines a few points at a time through round 15, and only while its group is ahead.
  - They fine 0.28–0.78 per member and round. The pool-maximisers fine 1.7–1.8.
  - They keep 4.5–4.6 members, as `zero` does.
  - Their own pool is 62.6–65.5 (`zero` 60.3). `ah`'s pool against them barely changes (46.7–49.8;
    46.1 against `zero`).
  - The leader instead fines 0.95 per member and round. That costs it 0.1 member but lifts its pool
    to 70.0, while `ah`'s pool rises only to 52.9.
- **Why the search missed it: the lead is too noisy at this sweep size.**
  - A design point's lead has a standard error of about 2.9 at 500 games. The whole range from
    `zero` to the leader is 2.9.
  - Sweeps also share their random streams by design index. Design point 245 plays the same games
    in every sweep, and on those games never punishing scores 20.83. Its true lead is 14.2.
  - In 7 of the 12 first sweeps, point 245 was a near-zero rule. In 6 of them it was the sweep's
    best point.
  - 9 of the 12 first final peaks collapsed to never punishing: exactly 16.76 on the shared final
    seed, the same games as `zero`.
- **Finals ran high.** The lineage bests' finals (18.62–19.28) are 2.6–3.6 above their fresh-seed
  leads. `zero` itself scores 16.76 on the final seed against 14.23 fresh.

## The objective

Per game, each group's common pool, 1.6·Σc − Σp per round, is averaged over the 24 rounds (an empty
group counts 0). The score is the rule's group's pool minus `ah`'s group's pool, averaged over games.
- **Seats:** the rule plays both seats, half the games each.
- **Games:** 250 per seat per sweep point, 1000 per seat per final point.
- **What agents knew:** they play `ah`, and the objective. They saw `ah`'s and `zero`'s reference sims,
  and from version 2 on their own lineage's results, with both pools.

## The campaign

- 12 fresh lineages, then 6 generations on the top 4: 36 versions, all scored.
- 4 h 40 min wall clock.
- $49.5 API-equivalent for the agents.

| Lineage | Gen 1 | Best final peak | Version |
|---|---|---|---|
| `-5` | 17.08 | 19.28 ± 1.50 | v7 |
| `-6` | 16.76 (`zero`) | 18.85 ± 1.45 | v6 |
| `-2` | 16.76 (`zero`) | 18.62 ± 1.46 | v7 |
| `-1` | 16.76 (`zero`) | 17.70 ± 1.44 | v6 |

- `-6`, `-2` and `-1` were picked for generation 2 among lineages tied at the `zero` value.
- The other eight lineages stayed at 16.14–16.76.
- From version 2 on, every lineage that iterated found something above `zero` on the final seed.

## Head to head on fresh seeds

Each manager against `ah`, in both seats, on 12 seeds no sweep or final used (20000–31000): 48,000
games each. Lead ± its standard error over games. The pools are per round. Fines are per member and
round of the manager's own group.

| Manager | Lead over `ah` | vs `zero` | Own pool | `ah`'s pool | Own members | Fines | `ah`'s fines |
|---|---|---|---|---|---|---|---|
| multi `-9-v5`, w = 0 | **17.11 ± 0.33** | +2.88 | 69.98 | 52.87 | 4.50 | 0.95 | 1.65 |
| `pf-20261112-6-v6` | 15.96 ± 0.29 | +1.73 | 62.68 | 46.72 | 4.59 | 0.28 | 1.75 |
| `pf-20261112-2-v7` | 15.88 ± 0.30 | +1.65 | 62.56 | 46.68 | 4.58 | 0.31 | 1.77 |
| `pf-20261112-5-v7` | 15.65 ± 0.31 | +1.42 | 65.48 | 49.83 | 4.52 | 0.78 | 1.69 |
| Cem's best | 15.08 ± 0.35 | +0.85 | 72.19 | 57.11 | 4.36 | 1.66 | 1.62 |
| `prev_best` (exp2) | 14.89 ± 0.35 | +0.66 | 72.92 | 58.03 | 4.37 | 1.83 | 1.58 |
| `zero` | 14.23 ± 0.29 | – | 60.28 | 46.05 | 4.58 | 0 | 1.76 |

- **Gaps between rows:** independent games, so the error of a gap is the two errors combined, about
  0.42–0.47. The leader is ahead of every new rule by 1.15–1.46 (2.6–3.3 se).
- **Seed by seed:** the leader is ahead of `-6-v6` on 9 of 12 seeds. Every new rule is ahead of
  `zero` on 10–11 of 12.
- **Earlier measurement of the leader:** `33_MULTI_h2h` (seeds 20000–25000) gave 18.34 ± 0.47. Here,
  on the same six seeds, it gives 17.14. The other pairings in each config shift the batch seeds, so
  these are different games. The difference, 1.2 ± 0.7, is within 2 se.

## What the new rules do

The three lineage bests reached similar ideas from different starts. They fine low givers
sparingly and never fine the rest.

- **`-6-v6`:** at round 4 only, every member who gave 1–5 points gets 21.4 points.
- **`-5-v7`:**
  - in round 0, the lowest contributor giving at most 6 gets 17.0;
  - in rounds 4 and 8, every member giving at most 6 gets 24.0 and 22.7.
- **`-2-v7`:**
  - fines 5.9 points to a member 4 or more below the others' mean;
  - only while its group leads `ah`'s on pool per member by at least 3.9;
  - only while it has at least 4 members;
  - only through round 15.

  It is the one rule that reads the other group: it uses `ah`'s pool to decide when fining is safe.

The agents' notes give the same account. Membership is the score, so fines that drive members out
cost twice: once in the rule's own pool and once again in `ah`'s, which gains them.

- In `-6-v6` and `-5-v7`, a hard one-off fine just after the first switch decision buys the low
  givers' contributions for the remaining 19 rounds.
- It comes when the next decision is three rounds away.
- Middling and high givers are never fined.

Their groups keep `zero`'s size (4.58–4.59 members). Their pool is 2–5 points above `zero`'s.

The leader (`-9-v5`, built for the own pool against four governors) fines more and longer.
- It fines newcomers and low givers through round 15.
- That costs it 0.08 member against `zero`.
- It lifts its own pool by 9.7 against `zero`.
- `ah`'s group gains only 6.8 of that back.

Its fines raise contributions more than they push players away. The lead search never got there.

## Why the search fell short

1. **Noise.**
   - A sweep point's lead has a standard error of 2.84–3.02 (500 games).
   - A pool's is about 2.1 at the same size. The lead is noisier because a member who leaves one
     group joins the other: both pools move.
   - Every rule here, `zero` included, lies within 3 points of the leader. The sweeps cannot rank
     points at that resolution.
   - The GP smooths this. Its peak goes where many points are mildly good, and near-zero rules are
     reliably mildly good.
2. **Shared random streams.**
   - Every sweep is seeded 42, and point i of every sweep plays the same batches.
   - Point 245's batches favour never punishing by 6.6 points: `zero`-like rules there score 20.83.
     Points 47, 7, 75, 127 and 241 are smaller cases.
   - In 7 of the 12 first sweeps, point 245 fell on a near-zero rule. In 6 of them it was the
     sweep's best point.
   - 9 of the 12 first finals collapsed to exactly the `zero` value.
   - The same seeding holds in every earlier campaign. It mattered here because the lead's true
     differences are small against its noise, and because "no punishment" is a point many rule
     families can reach.
3. **Local search.**
   - Generation 2 kept four lineages, three of them chosen among `zero` ties.
   - From there, each improved by small, targeted fines. None tried sustained moderate fining like
     the leader's.

## Follow-ups (not done)

- **Sweeps:** more games per sweep point for a difference objective, at least 2000 (lead se about
  1.4). Or a fresh seed per sweep, so that a lucky stream does not repeat across versions and
  lineages.
- **Seed a lineage from the leader** (`-9-v5`, w = 0) and let the lead search refine it.
- **Score the lead per member** (pool per member minus `ah`'s). This takes out the part of the lead
  that only counts heads.
