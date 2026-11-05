"""Sim (humanlike): marginal effect of small punishments, dummies per p level, by c bin; k=1 and k=4."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['run', 'episode', 'pid', 'round_number'])
d['c'] = np.where(d.contribution_valid, d.contribution, np.nan)
g = d.groupby(['run', 'episode', 'pid'])
for k in (1, 4):
    d[f'c{k}'] = g.c.shift(-k)
grp = d.groupby(['run', 'episode', 'round_number'])
d['gmean'] = grp.c.transform('mean')
d['others_p'] = (grp.punishment.transform('sum') - d.punishment) / 3
d['prev_c'] = g.c.shift(1)
h = d[d.run.str.contains('humanlike') & d.c.notna() & d.prev_c.notna()].copy()
h['pl'] = pd.cut(h.punishment, [-1, 0, 1, 2, 3, 5, 8, 12, 31], labels=['0', '1', '2', '3', '4-5', '6-8', '9-12', '13+'])
h['cb'] = pd.cut(h.c, [-1, 3, 7, 11, 15, 19, 20])
for k in (1, 4):
    print('k =', k, ' coefficient of p-level dummy vs p=0 (controls c, prev_c, gmean, others_p, round)')
    rows = {}
    for cb, s in h.groupby('cb'):
        s = s.dropna(subset=[f'c{k}'])
        m = smf.ols(f'c{k} ~ C(pl) + c + prev_c + gmean + others_p + round_number', data=s).fit()
        rows[str(cb)] = {lv: round(m.params.get(f'C(pl)[T.{lv}]', np.nan), 2) for lv in h.pl.cat.categories[1:]}
    print(pd.DataFrame(rows).T)
