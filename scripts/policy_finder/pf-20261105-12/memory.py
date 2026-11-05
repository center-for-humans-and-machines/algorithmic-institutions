"""Humanlike ref sim: does past punishment (beyond last round) still predict c_next? Does
punishing others hold up the high contributors?"""
import numpy as np, pandas as pd
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r = r[r.run.str.contains('humanlike')].copy()
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
r['c'] = np.where(r.contribution_valid, r.contribution, 0)
r = r.sort_values(['episode', 'player', 'round_number'])
gg = r.groupby(['episode', 'player'])
r['c_n1'] = gg.c.shift(-1); r['vn'] = gg.contribution_valid.shift(-1)
r['p_prev'] = gg.punishment.shift(1).fillna(0)
r['p_old'] = (gg.punishment.cumsum() - r.punishment - r.p_prev) / np.maximum(r.round_number - 1, 1)
key = ['episode', 'round_number']
r['oth'] = (r.groupby(key).c.transform('sum') - r.c) / 3
r['op'] = r.groupby(key).punishment.transform('sum') - r.punishment
s = r[r.contribution_valid & (r.vn == True) & (r.round_number >= 2)].dropna(subset=['c_n1'])
X = np.column_stack([np.ones(len(s)), s.c, s.oth, s.punishment, s.punishment > 0, s.p_prev, s.p_old, s.op])
print('c_n1 ~ 1,c,oth,p,p>0,p_prev,p_old_mean,others_p', np.linalg.lstsq(X, s.c_n1, rcond=None)[0].round(3))
h = s[(s.c >= 15) & (s.punishment == 0)]
print('high unpunished contributors: dc by others punishment bin')
print((h.c_n1 - h.c).groupby(pd.cut(h.op, [-1, 0, 3, 8, 20, 120])).agg(['mean', 'count']).round(2))
