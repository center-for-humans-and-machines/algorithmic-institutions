"""v2: unit check of the v2 rule on a hand-made history, and its static spend on the
zero-managed groups of the reference sims (contributions as they were, no response)."""
import torch as th, pandas as pd, numpy as np
from aimanager.manager.rule import load_rule, rule_inputs, run_rule
R = 'configs/managers/rule_based/pf-20261112-2-v2.yml'
code, prm = load_rule(R, {'thr': 3.0, 'fine': 4.0, 'w': 2, 'n_dec': 4})
# 8 players, group 0 = players 0-3 at first; player 0 always gives 0, player 1 gives 0 then 10,
# player 2 gives 15, player 3 gives 2 but times out in round 6; player 4 (other group) gives 0.
T = 24
c = th.zeros(1, 8, T); c[0, 1, 5:] = 10; c[0, 2] = 15; c[0, 3] = 2; c[0, 3, 6] = 0
valid = th.ones(1, 8, T); valid[0, 3, 6] = 0
grp = th.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 8, 1).expand(1, 8, T)
for t in range(T):
    inp = rule_inputs(c[..., :t + 1], valid[..., :t + 1], th.zeros(1, 8, t), grp[..., :t + 1], 0)
    p = run_rule(code, prm, inp)[0].tolist()
    print(t, [int(v) for v in p])
# static spend on reference zero groups
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(d.run.unique())
d = d[d.run.str.contains('zero_vs_ah') & (d.group_id == 0) | d.run.str.contains('ah_vs_zero') & (d.group_id == 1)].copy()
d['blk'] = d.round_number // 4
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
g = d.groupby(['run', 'episode', 'participant_code', 'blk'])
d['bm'] = g.contribution.cumsum() / (g.cumcount() + 1)  # same-group block mean (players stay within a block)
dec = d[d.round_number.isin([3, 7, 11, 15, 19])]
for thr in [1, 2, 3, 4, 6, 8, 10]:
    fr = ((dec.bm < thr) & (dec.contribution < thr)).mean()
    print(f'thr {thr}: share of members flagged at a decision round {fr:.3f}; members/group {dec.groupby(["run","episode","round_number"]).size().mean():.2f}')
