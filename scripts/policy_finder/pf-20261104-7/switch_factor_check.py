"""v1 sweep: switch_factor was randomised, so its effect on my group's size after each switch and on
the pool is causal. Per point: mean size at rounds 4, 8, 12 and pool; then binned by switch_factor,
within points that actually punish (rate*threshold large)."""
import json
import numpy as np
import pandas as pd
pd.set_option('display.width', 250)
B = 'plots/simulation/policy_finder/pf-20261104-7_sweep/'
sw = json.load(open(B + 'sweep.json'))
P = pd.DataFrame([dict(p['params'], name=p['name'], pool=p['pool']) for p in sw['points']]).set_index('name')
rows = []
for part in ['p1of2', 'p2of2']:
    d = pd.read_parquet(B + f'per_round_{part}.parquet', columns=['run', 'episode', 'round_number', 'agent_group', 'punishment'],
                        filters=[('round_number', 'in', [3, 4, 8, 12])])
    d['run'] = d.run.astype('category')
    neps = d.groupby('run', observed=True).episode.nunique()
    g0 = d[d.agent_group == 0]
    n = g0.groupby(['run', 'round_number'], observed=True).size().unstack() .div(neps, axis=0)
    p3 = g0[g0.round_number == 3].groupby('run', observed=True).punishment.mean()
    n['p3'] = p3
    rows.append(n)
N = pd.concat(rows)
N.index = [r.split('managed by ')[1].replace('_vs_ah', '') if 'managed by' in r else r for r in N.index]
P = P.join(N)
P['strength'] = P.rate * P.threshold
for lab, Q in [('all', P), ('rate*threshold>8 and t_stop>4', P[(P.strength > 8) & (P.t_stop > 4)])]:
    Q = Q.assign(sf=pd.cut(Q.switch_factor, [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True))
    print(lab); print(Q.groupby('sf', observed=True)[['p3', 4, 8, 12, 'pool']].mean().round(2), Q.groupby('sf', observed=True).size().to_dict())
    X = np.column_stack([np.ones(len(Q)), Q.switch_factor, np.log(Q.rate), Q.threshold, Q.t_stop])
    for y in [4, 'pool']:
        b, *_ = np.linalg.lstsq(X, Q[y].values, rcond=None); r = Q[y].values - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var())
        print(' ', y, 'switch_factor coef', round(b[1], 3), 'se', round(se[1], 3))
