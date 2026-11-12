"""v4: do sweep points with the same index share seeds across versions? Then the paired effect
of every fining point against a never-fining point at the same index."""
import pandas as pd, numpy as np
G = pd.read_parquet('/tmp/v4_games.parquet')
sp = G.groupby(['ver', 'idx']).spend.sum().unstack('ver')
zero = sp == 0
print('never-fining points per version', zero.sum().to_dict())
W = G.set_index(['ver', 'idx', 'seat', 'episode']).lead.unstack('ver')
both = zero.index[(zero.sum(axis=1) >= 2)]
for i in both[:5]:
    vs = zero.columns[zero.loc[i]]
    x = W.loc[i][vs]
    print(i, list(vs), 'max abs diff', (x.iloc[:, 0] - x.iloc[:, 1]).abs().max().round(4))
# paired effects
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
P.to_parquet('/tmp/v4_pairs.parquet')
s = P.groupby(['ver', 'idx']).agg(d=('d', 'mean'), se=('d', 'sem'), spend=('spend', 'mean'), fired=('t_first', lambda s: s.notna().mean()), dn=('dn', 'mean')).reset_index()
print(s.round(2).to_string())
print('overall per version', s.groupby('ver')[['d', 'spend', 'dn']].mean().round(2))
