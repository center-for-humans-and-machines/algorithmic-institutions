"""v2: personal vs shared effect. Late (12-23) contribution anywhere, by round-0 type and
starting group, v1 peak vs zero ref; and regression of a player's late contribution on own
punishment received in rounds 0-11 and the punishment others in its group received."""
import pandas as pd, numpy as np
from numpy.linalg import lstsq
F = 'plots/simulation/policy_finder/pf-20261104-5_sweep/final/per_round.parquet'
R = 'plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'
d = pd.read_parquet(F)
r = pd.read_parquet(R); r = r[r.run.str.contains('zero_vs_ah')]
k = ['run', 'episode', 'participant_code']
for name, x in [('v1 peak', d[d.run.str.contains('_gp_')]), ('zero', r), ('all v1 final', d)]:
    x = x.copy()
    f0 = x[x.round_number == 0].set_index(k)
    x = x.join(pd.cut(f0.contribution.where(f0.contribution_valid), [-1, 4, 9, 14, 20]).rename('typ'), on=k)
    x = x.join(f0.group_id.rename('g0'), on=k)
    late = x[(x.round_number >= 12) & x.contribution_valid]
    print('==', name); print(late.pivot_table(index='typ', columns='g0', values='contribution', aggfunc='mean', observed=True).round(2))
    e = x[x.round_number < 12].copy()
    e['own_p0'] = np.where(e.group_id == 0, e.punishment, 0)
    gsum = e.groupby(['run', 'episode', 'round_number', 'group_id']).punishment.transform('sum')
    gn = e.groupby(['run', 'episode', 'round_number', 'group_id']).punishment.transform('size')
    e['oth'] = (gsum - e.punishment) / (gn - 1).clip(lower=1)
    pl = e.groupby(k).agg(own=('punishment', 'mean'), oth=('oth', 'mean'), c_e=('contribution', 'mean'))
    pl = pl.join(late.groupby(k).contribution.mean().rename('cl')).join(f0.contribution.rename('c0')).dropna()
    X = np.c_[np.ones(len(pl)), pl.c0, pl.own, pl.oth]
    print('late c ~ 1 + c0 + own pun/round(0-11) + others-in-group pun/round(0-11):', lstsq(X, pl.cl.values, rcond=None)[0].round(3), 'n', len(pl))
