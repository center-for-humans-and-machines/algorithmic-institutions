"""Run the rule on zero-managed groups of the reference sims (their histories) to see how many
members it would flag per round for various thr/persist, and test its semantics on a toy history."""
import pandas as pd, numpy as np, torch as th
from aimanager.manager.rule import read_rule, run_rule
rule, code = read_rule('configs/managers/rule_based/pf-20261112-7.yml', 'exp2_diff')
# toy: player 0 gives 0 for 3 rounds in group; player 1 gives 0 but was out of group in round 0
c = th.tensor([[[0, 0, 0], [0, 0, 0], [10, 10, 10]] + [[10, 10, 10]] * 5], dtype=th.float)
ing = th.ones_like(c); ing[0, 1, 0] = 0; ing[0, 4:, :] = 0
inp = dict(c=c, valid=th.ones_like(c), p=th.zeros(1, 8, 2), in_group=ing, t=th.tensor(2.))
for pers in (1, 2, 3):
    print(pers, run_rule(code, dict(thr=1., amount=3., persist=pers, t_stop=23), inp).tolist())
print('t_stop=1', run_rule(code, dict(thr=1., amount=3., persist=1, t_stop=1), inp).tolist())
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run == 'ah group_switching managed by zero_vs_zero']
d = d.sort_values(['episode', 'participant_code', 'round_number'])
g = d.groupby(['episode', 'participant_code'])
for thr in (0, 2, 5, 8):
    low = (d.contribution <= thr) & d.contribution_valid.astype(bool)
    run = low.astype(int); same = (g.agent_group.shift(1) == d.agent_group)
    s1 = low; s2 = low & g.contribution.shift(1).le(thr) & same
    s3 = s2 & g.contribution.shift(2).le(thr) & (g.agent_group.shift(2) == d.agent_group)
    print('thr', thr, 'share flagged persist1 %.3f persist2 %.3f persist3 %.3f' % (s1.mean(), s2.mean(), s3.mean()))
