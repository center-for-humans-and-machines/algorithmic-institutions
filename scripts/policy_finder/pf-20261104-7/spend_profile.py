"""v1+v2 sweeps, per design point: punishment spent per member in rounds 0-3, 4-11, 12-23; my group's
size after the first switch (round 4) and late (12-23); contribution late; pool. Then OLS of the
pool and of the sizes on the three spends (with squares), to see where punishment pays."""
import json
import numpy as np
import pandas as pd
pd.set_option('display.width', 250)
B = 'plots/simulation/policy_finder/'
out = []
for v in ['pf-20261104-7', 'pf-20261104-7-v2']:
    sw = json.load(open(f'{B}{v}_sweep/sweep.json'))
    P = pd.DataFrame([dict(name=p['name'], pool=p['pool'], members=p['members']) for p in sw['points']]).set_index('name')
    rows = []
    for part in ['p1of2', 'p2of2']:
        d = pd.read_parquet(f'{B}{v}_sweep/per_round_{part}.parquet', columns=['run', 'episode', 'round_number', 'agent_group', 'punishment', 'contribution'])
        d = d[d.agent_group == 0]
        d['run'] = d.run.astype('category')
        d['blk'] = np.select([d.round_number < 4, d.round_number < 12], ['e', 'm'], 'l')
        neps = d.groupby('run', observed=True).episode.nunique()
        sp = d.groupby(['run', 'blk'], observed=True).agg(p=('punishment', 'mean'), c=('contribution', 'mean'), n=('punishment', 'size')).unstack()
        sp.columns = [f'{a}_{b}' for a, b in sp.columns]
        for b, k in [('e', 16), ('m', 32), ('l', 48)]:
            sp[f'n_{b}'] = sp[f'n_{b}'] / neps / k * 4 / 4
        sp['n4'] = d[d.round_number == 4].groupby('run', observed=True).size() / neps
        rows.append(sp); del d
    S = pd.concat(rows)
    S.index = [r.split('managed by ')[1].replace('_vs_ah', '') for r in S.index]
    S = S.join(P); S['ver'] = v
    out.append(S)
S = pd.concat(out)
S.to_csv('/tmp/spend_profile.csv')
print(S.sort_values('pool', ascending=False).head(15).round(2).to_string())
X = np.column_stack([np.ones(len(S)), S.p_e, S.p_m, S.p_l, S.p_e ** 2, S.p_m ** 2, S.p_l ** 2, S.p_e * S.p_m])
names = ['1', 'pe', 'pm', 'pl', 'pe2', 'pm2', 'pl2', 'pe*pm']
for y in ['pool', 'n4', 'n_l', 'c_l']:
    b, *_ = np.linalg.lstsq(X, S[y].values, rcond=None); r = S[y].values - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var())
    print(y, 'R2', round(1 - r.var() / S[y].var(), 3)); print(pd.DataFrame({'b': b, 'se': se}, index=names).round(3).T.to_string())
