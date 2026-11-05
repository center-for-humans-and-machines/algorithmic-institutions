"""Artificial humans in the reference sims: response to punishment, timeouts, per-run totals."""
import pandas as pd, numpy as np, numpy.linalg as la
pd.set_option('display.width',200)
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(r.columns.tolist(), r.shape)
r['pid'] = r.participant_code.str.split('_').str[0].astype(int)
r['c'] = np.where(r.contribution_valid, r.contribution, 0)
# score: per round mean payoff of valid, summed over rounds, averaged over episodes
v = r[r.contribution_valid]
sc = v.groupby(['run','episode','round_number']).payoff.mean().groupby(['run','episode']).sum().groupby('run').agg(['mean','sem'])
print(sc)
print('timeout rate', 1-r.groupby('run').contribution_valid.mean())
r = r.sort_values(['run','episode','pid','round_number'])
g = r.groupby(['run','episode','pid'])
r['c_next'] = g.c.shift(-1); r['v_next'] = g.contribution_valid.shift(-1)
r['gsum'] = r.groupby(['run','episode','round_number']).c.transform('sum')
r['nv'] = r.groupby(['run','episode','round_number']).contribution_valid.transform('sum')
r['others'] = (r.gsum - r.c)/(r.nv - 1).clip(lower=1)
d = r[r.contribution_valid & (r.v_next==True)].copy()
d['dc'] = d.c_next - d.c
for run, e in d.groupby('run'):
    X = np.column_stack([np.ones(len(e)), e.punishment, e.c, e.others])
    print(run, 'c_next ~ 1,p,c,others', la.lstsq(X, e.c_next, rcond=None)[0].round(3))
h = d[d.run.str.contains('humanlike')].copy()
h['pbin'] = pd.cut(h.punishment, [-1,0,2,5,10,30]); h['cbin'] = pd.cut(h.c, [-1,4,9,14,19,20])
print(h.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean', observed=False).round(2))
print(h.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='count', observed=False))
z = d[d.run.str.contains('zero')].copy(); z['cbin']=pd.cut(z.c,[-1,4,9,14,19,20])
print('zero dc by cbin', z.groupby('cbin',observed=False).dc.mean().round(2))
