"""v2: is v1's late gain a group norm (herding) or a personal memory of punishment?
Late (rounds 12-23) contributions in each group, and of players never punished by our manager,
split by how many rounds they spent in our group before round 12."""
import pandas as pd, numpy as np
F = 'plots/simulation/policy_finder/pf-20261104-5_sweep/final/per_round.parquet'
R = 'plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'
d = pd.read_parquet(F); d = d[d.run.str.contains('_gp_')]
r = pd.read_parquet(R); r = r[r.run.str.contains('zero_vs_ah')]
k = ['episode', 'participant_code']
for name, x in [('v1 peak', d), ('zero', r)]:
    x = x.copy()
    late = x[(x.round_number >= 12) & x.contribution_valid]
    print('==', name, 'late mean c by group:', late.groupby('group_id').contribution.mean().round(2).to_dict())
    e = x[x.round_number < 12]
    ours = e[e.group_id == 0].groupby(k).agg(r0=('round_number', 'size'), p0=('punishment', 'sum'))
    pah = e[e.group_id == 1].groupby(k).punishment.sum().rename('pah')
    pl = ours.reindex(x.groupby(k).size().index).fillna(0).join(pah).fillna(0)
    L = late[late.group_id == 0].join(pl, on=k)
    L['r0b'] = pd.cut(L.r0, [-1, 0, 4, 8, 12])
    L['p0b'] = pd.cut(L.p0, [-1, 0, 20, 60, 1000])
    print('late c in our group by rounds spent with us before 12 x our punishment received before 12:')
    print(L.pivot_table(index='r0b', columns='p0b', values='contribution', aggfunc='mean', observed=True).round(2))
    print(L.pivot_table(index='r0b', columns='p0b', values='contribution', aggfunc='size', observed=True))
    # trajectory of mean contribution per round in each group
    print(x[x.contribution_valid].groupby(['group_id', 'round_number']).contribution.mean().unstack().round(1).iloc[:, ::2])
