"""v6: v4's sweep points (relative shirker without a race condition) by the per-game baseline, to
compare with v5's low-gap region: is the lead condition what made v5's low gap positive?"""
import pandas as pd, numpy as np, json
pt = pd.read_csv('scripts/policy_finder/pf-20261112-2/v6_gamebase_points.csv')
v = json.load(open('plots/simulation/policy_finder/pf-20261112-2-v4_sweep/sweep.json'))
par = pd.DataFrame([dict(idx=int(p['name'].split('_s')[-1]), **p['params']) for p in v['points']])
s = pt[pt.ver == 'pf-20261112-2-v4'].merge(par, on='idx')
s['se_all'] = s.se_fired.fillna(0) * s.fired
on = s[s.fine >= 1]
def b(x):
    return pd.Series({'n': len(x), 'd_all': x.d_all.mean(), 'se': np.sqrt((x.se_all ** 2).sum()) / len(x),
                      'd_fired': x.d_fired.mean(), 'fired': x.fired.mean(), 'spend': x.spend.mean(), 'cov': x['cov'].mean()})
print('v4 fining points', len(on)); print(b(on).round(3).to_dict())
print(on.groupby([pd.cut(on.gap, [5, 8, 11, 15]), pd.cut(on.fine, [1, 4, 7, 10])], observed=True).apply(b, include_groups=False).round(3).to_string())
print(on.groupby([pd.cut(on.gap, [5, 8, 11, 15]), pd.cut(on.last_block, [2, 4, 6])], observed=True).apply(b, include_groups=False).round(3).to_string())
lo = on[on.gap < 8]
for c in ['n_min', 'start']:
    print(c); print(lo.groupby(c).apply(b, include_groups=False).round(3).to_string())
