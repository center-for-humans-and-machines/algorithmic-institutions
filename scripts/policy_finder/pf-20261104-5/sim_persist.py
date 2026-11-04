"""Ref sims: does the contribution response to a punishment persist over h rounds?
Regress c_{t+h}-c_t on c_t, p_t, group mean c_t for low contributors (c_t<10) who stay
in the ah-managed group; also the same in the human data."""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['mgr'] = [r.split('managed by ')[1].split('_vs_')[g] for r, g in zip(d.run, d.group_id)]
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
k = ['run', 'episode', 'participant_code']
d['gm'] = d.groupby(['run', 'episode', 'round_number', 'group_id']).contribution.transform('mean')
def est(d, k, label):
    for h in [1, 2, 3, 4, 6]:
        cn = d.groupby(k).contribution.shift(-h)
        gn = d.groupby(k).group_id.shift(-h)
        m = (gn == d.group_id) & cn.notna() & (d.contribution < 10) & (d.round_number + h <= 23)
        s = d[m]
        X = np.c_[np.ones(len(s)), s.contribution, s.punishment, s.gm]
        b = lstsq(X, (cn[m] - s.contribution).values, rcond=None)[0]
        print(f'{label} h={h} n={len(s)} slope on p {b[2]:.3f}')
est(d[d.mgr == 'ah'], k, 'sim')
h = pd.read_pickle('/tmp/pf5_human.pkl').rename(columns={})
h['gm'] = h.gmean
est(h, ['session', 'episode', 'participant_code'], 'human')
