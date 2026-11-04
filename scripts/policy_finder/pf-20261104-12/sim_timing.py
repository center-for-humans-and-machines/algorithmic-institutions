"""Reference sims, ah groups: switching vs punishment at each position of the block."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[d.run.str.contains('ah_vs')].copy()
d['mgr'] = np.where(d.run.str.contains('ah_vs_ah') | ((d.agent_group==0) == d.run.str.contains('by ah_vs')), 'ah', 'zero')
d = d.sort_values(['run','episode','participant_code','round_number'])
key = ['run','episode','participant_code']
g = d.groupby(key)
d['switched'] = (g.agent_group.shift(-1) != d.agent_group).astype(float)
d['pos'] = d.round_number % 4; d['block'] = d.round_number // 4
gp = d.groupby(['run','episode','round_number','agent_group']).punishment.transform('mean')
d['gp_others'] = gp
w = d.pivot_table(index=key+['block'], columns='pos', values=['punishment','contribution','gp_others'])
w.columns = [f'{a}{b}' for a,b in w.columns]
dec = d[d.pos==3].set_index(key+['block'])[['switched','mgr','agent_group']].join(w).reset_index()
dec = dec[(dec.mgr=='ah') & (dec.block<=4)].dropna()
X = dec[[f'punishment{i}' for i in range(4)] + [f'gp_others{i}' for i in range(4)] + ['contribution3']].copy()
X['first'] = (dec.block==0).astype(float)
m = LogisticRegression(max_iter=1000).fit(X, dec.switched)
for n, cf in zip(X.columns, m.coef_[0]): print(f'{n:15s} {cf:+.3f}')
print('intercept', m.intercept_)
