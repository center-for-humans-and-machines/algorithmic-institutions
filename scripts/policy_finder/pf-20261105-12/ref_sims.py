"""Reference sims: scores, timeouts, punishment pattern of humanlike, response of next contribution to punishment."""
import sys
import numpy as np, pandas as pd
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
r['c'] = np.where(r.contribution_valid, r.contribution, 0)
for run, g in r.groupby('run'):
    v = g[g.contribution_valid]
    per = v.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum()
    print(run, 'score %.1f se %.1f' % (per.mean(), per.std() / np.sqrt(len(per))),
          'timeout rate %.3f' % (1 - g.contribution_valid.mean()),
          'mean c valid %.2f' % v.contribution.mean(), 'mean p %.2f' % v.punishment.mean())
    print('  c by round', v.groupby('round_number').contribution.mean().round(1).values)
# humanlike: what punishment given contribution
h = r[r.run.str.contains('humanlike') & r.contribution_valid]
print('humanlike p by c bin'); print(h.groupby(pd.cut(h.contribution, [-1, 0, 5, 10, 14, 19, 20])).punishment.agg(['mean', 'count']).round(2))
print('frac punished', (h.punishment > 0).mean())
# next-round response
for run, g in r.groupby('run'):
    g = g.sort_values(['episode', 'player', 'round_number']).copy()
    gg = g.groupby(['episode', 'player'])
    g['c_next'] = gg.c.shift(-1); g['valid_next'] = gg.contribution_valid.shift(-1)
    others = g.groupby(['episode', 'round_number']).c.transform('sum') - g.c
    nv = g.groupby(['episode', 'round_number']).contribution_valid.transform('sum') - g.contribution_valid
    g['oth'] = others / nv.clip(lower=1)
    s = g[g.contribution_valid & (g.valid_next == True)].dropna(subset=['c_next'])
    s['dc'] = s.c_next - s.c
    print(run, 'dc by p bin, c bin')
    print(s.groupby([pd.cut(s.c, [-1, 5, 10, 15, 20]), pd.cut(s.punishment, [-1, 0, 2, 5, 10, 30])], observed=True).dc.agg(['mean', 'count']).round(2).unstack())
    X = np.column_stack([np.ones(len(s)), s.c, s.punishment, s.oth, s.round_number])
    b = np.linalg.lstsq(X, s.c_next, rcond=None)[0]
    print('  c_next ~ 1,c,p,oth,round:', b.round(3))
