"""v5: paired per-game effects (fining point vs never-fining point at the same sweep index, any
of v1-v4), then the effect by the round of the game's first fine (a pre-treatment cut)."""
import pandas as pd, numpy as np
G = pd.read_parquet('/tmp/v5_games.parquet')
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
P.to_parquet('/tmp/v5_pairs.parquet')
print('pairs', P.groupby('ver').idx.nunique().to_dict())
s = P.groupby(['ver', 'idx']).agg(d=('d', 'mean'), se=('d', 'sem'), spend=('spend', 'mean')).reset_index()
print('per version mean paired d / spend', s.groupby('ver')[['d', 'spend']].mean().round(2).to_dict())
print(s[s.ver == 'pf-20261112-2-v4'].sort_values('d').round(2).to_string())
f = P[P.t_first.notna()].copy()
f['tb'] = pd.cut(f.t_first, [-1, 3, 7, 11, 15, 19, 20, 21, 22, 23])
f['ph'] = f.t_first % 4
def agg(x):
    return pd.Series({'n': len(x), 'd': x.d.mean(), 'se': x.d.sem(), 'spend': x.spend.mean(),
                      'per_pt': x.d.sum() / x.spend.sum() * 24, 'dn': x.dn.mean()})
print(f.groupby('tb').apply(agg).round(3))
print(f[f.t_first < 20].groupby('ph').apply(agg).round(3))
print(f.groupby(['ver', 'tb']).apply(agg).round(3).to_string())
