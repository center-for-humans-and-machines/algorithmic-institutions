"""Reference sims: score per manager, and how next contribution responds to punishment."""
import pandas as pd, numpy as np
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
v = r[r.contribution_valid]
score = v.groupby(['run', 'episode', 'round_number']).payoff.mean().groupby(['run', 'episode']).sum()
print(score.groupby('run').agg(['mean', 'sem']))
r = r.sort_values(['run', 'episode', 'player', 'round_number'])
g = r.groupby(['run', 'episode', 'player'])
r['c_next'] = g.contribution.shift(-1)
r['valid_next'] = g.contribution_valid.shift(-1)
grp = r[r.contribution_valid].groupby(['run', 'episode', 'round_number'])
r['gmean'] = grp.contribution.transform('mean')
r['others_p'] = grp.punishment.transform('sum') - r.punishment
h = r[(r.run.str.contains('humanlike')) & r.contribution_valid & (r.valid_next == True)].copy()
h['dc'] = h.c_next - h.contribution
h['pbin'] = pd.cut(h.punishment, [-1, 0, 2, 4, 6, 9, 14, 30])
h['cbin'] = pd.cut(h.contribution, [-1, 4, 9, 14, 19, 20])
print(h.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(h.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size'))
# does punishing others matter for an unpunished player?
u = h[h.punishment == 0].copy()
u['opbin'] = pd.cut(u.others_p, [-1, 0, 3, 8, 15, 90])
print(u.pivot_table(index='cbin', columns='opbin', values='dc', aggfunc='mean').round(2))
# zero run: dc by contribution
z = r[(r.run.str.contains('zero')) & r.contribution_valid & (r.valid_next == True)].copy()
z['dc'] = z.c_next - z.contribution
z['cbin'] = pd.cut(z.contribution, [-1, 4, 9, 14, 19, 20])
z['rel'] = pd.cut(z.contribution - z.gmean, [-21, -5, -1, 1, 5, 21])
print(z.pivot_table(index='cbin', columns='rel', values='dc', aggfunc='mean').round(2))
h['rel'] = pd.cut(h.contribution - h.gmean, [-21, -5, -1, 1, 5, 21])
print(h.pivot_table(index='rel', columns='pbin', values='dc', aggfunc='mean').round(2))
