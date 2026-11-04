"""Reference sims: does a punishment's effect persist (c_{t+k}-c_t) and spill over onto others?"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs_ah')]
d = d.sort_values(['run','episode','participant_code','round_number'])
g = d.groupby(['run','episode','participant_code'])
for k in (1,2,3): d[f'c{k}'] = g.contribution.shift(-k); d[f'g{k}'] = g.agent_group.shift(-k)
# others' punishment in same group-round
gs = d.groupby(['run','episode','round_number','agent_group'])
d['psum'] = gs.punishment.transform('sum'); d['n'] = gs.punishment.transform('count')
d['p_oth'] = (d.psum-d.punishment)/(d.n-1).clip(lower=1)
d['gc'] = gs.contribution.transform('mean')
v = d[(d.round_number%4==0)&(d.g3==d.agent_group)&(d.contribution_valid==1)]  # block start, 3 rounds ahead stay in block
v['pbin']=pd.cut(v.punishment,[-1,0,3,6,10,30]); v['cbin']=pd.cut(v.contribution,[-1,4,9,14,20])
for k in (1,2,3):
    v[f'd{k}']=v[f'c{k}']-v.contribution
    print(f'c(t+{k})-c(t), block-start rounds'); print(v.pivot_table(index='cbin',columns='pbin',values=f'd{k}',aggfunc='mean',observed=True).round(2))
w = d[(d.g1==d.agent_group)&(d.contribution_valid==1)&(d.round_number%4!=3)].dropna(subset=['c1'])
X = np.c_[np.ones(len(w)), w.contribution, w.gc, np.sqrt(w.punishment), np.sqrt(w.p_oth), w.round_number]
print('dc ~ 1, c, gmean c, sqrt p_own, sqrt mean p_others, t:', np.linalg.lstsq(X, w.c1-w.contribution, rcond=None)[0].round(3))
