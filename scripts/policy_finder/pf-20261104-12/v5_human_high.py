"""v5: human games - which managers' groups reach high contributions late, and how they punished
(per group-episode: late mean contribution, punishment per member by block, share of members punished,
punishment of members by own contribution band)."""
import pandas as pd, numpy as np
d = pd.read_csv('experiments/2group_8agent_50ep.csv')
d = d.drop_duplicates(['session','episode','round_number','participant_code'])
d['block'] = d.round_number // 4
d['ge'] = d.session.astype(str) + '_' + d.episode.astype(str) + '_' + d.group_id.astype(str)
rows = []
for ge, x in d.groupby('ge'):
    r = {'ge': ge}
    r['c_late'] = x[x.round_number >= 12].contribution.mean()
    r['c_b0'] = x[x.block == 0].contribution.mean()
    r['n_late'] = x[x.round_number >= 12].groupby('round_number').size().mean()
    for b in range(6):
        xb = x[x.block == b]
        r[f'p{b}'] = xb.punishment.mean()
    r['p_low'] = x[x.contribution <= 5].punishment.mean()
    r['p_mid'] = x[(x.contribution > 5) & (x.contribution <= 12)].punishment.mean()
    r['p_hi'] = x[x.contribution > 12].punishment.mean()
    r['pool'] = x.groupby('round_number').apply(lambda y: 1.6*y.contribution.sum() - y.punishment.sum()).reindex(range(24), fill_value=0).mean()
    rows.append(r)
g = pd.DataFrame(rows)
g['q'] = pd.qcut(g.c_late, 4, labels=False)
pd.set_option('display.width', 200)
print(g.groupby('q').mean(numeric_only=True).round(2).to_string())
print(g[['pool','c_late','c_b0','n_late','p0','p1','p2','p3','p_low','p_mid','p_hi']].corr().round(2).loc[['pool','c_late','n_late']].to_string())
