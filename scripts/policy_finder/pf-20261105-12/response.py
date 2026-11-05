"""Humanlike ref sim: next-round contribution change by (c, p), own and others' (group-level) response."""
import numpy as np, pandas as pd
pd.set_option('display.width', 200)
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
r['c'] = np.where(r.contribution_valid, r.contribution, 0)
g = r[r.run.str.contains('humanlike')].sort_values(['episode', 'player', 'round_number']).copy()
gg = g.groupby(['episode', 'player'])
for k in (1, 2, 4):
    g[f'c_n{k}'] = gg.c.shift(-k)
g['vn'] = gg.contribution_valid.shift(-1)
key = ['episode', 'round_number']
g['P'] = g.groupby(key).punishment.transform('sum')
g['op'] = g.P - g.punishment
g['nv'] = g.groupby(key).contribution_valid.transform('sum')
g['oth'] = (g.groupby(key).c.transform('sum') - g.c) / (g.nv - g.contribution_valid).clip(lower=1)
s = g[g.contribution_valid & (g.vn == True)].dropna(subset=['c_n1']).copy()
s['dc'] = s.c_n1 - s.c
pb = pd.cut(s.punishment, [-1, 0, 1, 2, 3, 5, 8, 12, 30])
cb = pd.cut(s.c, [-1, 0, 4, 8, 12, 16, 19, 20])
print('mean dc next round'); print(s.groupby([cb, pb], observed=True).dc.mean().round(2).unstack())
print('count'); print(s.groupby([cb, pb], observed=True).dc.size().unstack())
# with others' punishment and other's contribution controls
X = np.column_stack([np.ones(len(s)), s.c, s.oth, s.punishment, np.minimum(s.punishment, 3), s.op, s.round_number])
for k in (1, 2, 4):
    m = s[f'c_n{k}'].notna().values
    b = np.linalg.lstsq(X[m], s[f'c_n{k}'].values[m], rcond=None)[0]
    print(f'c_t+{k} ~ 1,c,oth,p,min(p,3),others_p,round', b.round(3))
