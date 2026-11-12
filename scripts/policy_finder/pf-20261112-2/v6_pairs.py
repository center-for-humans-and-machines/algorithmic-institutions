"""v6: v5_pairs with v5's sweep added; then, for v5's fining points, the paired effect per point
against its params (point-level regressions and bins), since v5's raw landscape was all seed noise."""
import pandas as pd, numpy as np, json
G = pd.read_parquet('/tmp/v6_games.parquet')
sp = G.groupby(['ver', 'idx']).spend.sum().unstack('ver')
zero = sp == 0
print('never-fining points per version', zero.sum().to_dict(), 'indices with one', zero.any(axis=1).sum())
F = G.set_index(['ver', 'idx', 'seat', 'episode'])
pairs = []
for i in zero.index[zero.any(axis=1)]:
    base = zero.columns[zero.loc[i]][0]
    for v in zero.columns[~zero.loc[i]]:
        a = F.loc[(base, i)]; b = F.loc[(v, i)]
        j = b[['lead', 'spend', 't_first', 'n_end']].join(a[['lead', 'n_end']], rsuffix='_0')
        j['d'] = j.lead - j.lead_0; j['dn'] = j.n_end - j.n_end_0
        j['ver'] = v; j['idx'] = i
        pairs.append(j.reset_index())
P = pd.concat(pairs)
P.to_parquet('/tmp/v6_pairs.parquet')
s = P.groupby(['ver', 'idx']).agg(d=('d', 'mean'), se=('d', 'sem'), spend=('spend', 'mean'),
                                  fired=('t_first', lambda x: x.notna().mean()), dn=('dn', 'mean')).reset_index()
print('per version mean paired d / spend', s.groupby('ver')[['d', 'spend']].mean().round(2).to_string())
v5 = json.load(open('plots/simulation/policy_finder/pf-20261112-2-v5_sweep/sweep.json'))
par = pd.DataFrame([dict(idx=int(p['name'].split('_s')[-1]), **p['params']) for p in v5['points']])
s5 = s[s.ver == 'pf-20261112-2-v5'].merge(par, on='idx')
s5.to_csv('scripts/policy_finder/pf-20261112-2/v6_v5_paired_points.csv', index=False)
print('v5 paired fining points', len(s5), 'mean d %.3f, se of mean %.3f' % (s5.d.mean(), s5.d.std() / np.sqrt(len(s5))))
print(s5.sort_values('d').round(2).to_string())
for c, bins in [('lead', [0, 5, 10, 15, 20, 25]), ('gap', [5, 7, 9, 11, 13, 15]), ('fine', [1, 2, 3, 4, 6]), ('last_block', [0, 1, 2, 3, 4, 5, 6])]:
    g = s5.groupby(pd.cut(s5[c], bins), observed=True).agg(n=('d', 'size'), d=('d', 'mean'), se=('se', lambda x: np.sqrt((x ** 2).sum()) / len(x)), spend=('spend', 'mean'), fired=('fired', 'mean'))
    print(c); print(g.round(3).to_string())
X = np.c_[np.ones(len(s5)), s5.lead, s5.gap, s5.fine, s5.last_block]
w = 1 / s5.se ** 2
b = np.linalg.lstsq(X * np.sqrt(w.values)[:, None], s5.d * np.sqrt(w), rcond=None)[0]
print('WLS d ~ 1 + lead + gap + fine + last_block', b.round(3))
