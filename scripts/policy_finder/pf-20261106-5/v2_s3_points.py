# v2: per point x opponent: realised punishment per member-round by phase (rounds 0-3,4-7,8-15,16-23),
# for my group and the opponent's; relate my pool to my punishment level.
import pandas as pd, numpy as np
a = pd.read_csv('/tmp/v1agg.csv')
tot = a.groupby(['pt', 'opp']).n.sum()
neps = tot / (8 * 24)
a['ppm'] = a.p / a.n
a['cpm'] = a.c / a.n
a['members'] = a.n / a.set_index(['pt', 'opp']).index.map(neps).values / np.array([4, 4, 8, 8])[a.phase]
a['pool'] = (1.6 * a.c - a.p) / a.set_index(['pt', 'opp']).index.map(neps).values / np.array([4, 4, 8, 8])[a.phase]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
# opponents' own behaviour (opponent group), averaged over all points
print('opponent group: punishment/member, contribution/member, members, pool by phase')
print(a[a.mine == 0].groupby(['opp', 'phase'])[['ppm', 'cpm', 'members', 'pool']].mean().round(2).unstack())
print('my group, mean over points')
print(a[a.mine == 1].groupby(['opp', 'phase'])[['ppm', 'cpm', 'members', 'pool']].mean().round(2).unstack())
m = a[a.mine == 1].pivot_table(index=['pt', 'opp'], columns='phase', values=['ppm', 'pool', 'members', 'cpm'])
m.columns = [f'{x}{y}' for x, y in m.columns]
m = m.reset_index()
m['P'] = (m.pool0 * 4 + m.pool1 * 4 + m.pool2 * 8 + m.pool3 * 8) / 24
m['pE'] = (m.ppm0 + m.ppm1) / 2
m['pL'] = (m.ppm2 + m.ppm3) / 2
for o, g in m.groupby('opp'):
    g = g.copy()
    g['bE'] = pd.cut(g.pE, [-1, 0.05, 0.3, 1, 2, 4, 30])
    print(o)
    print(g.groupby('bE')[['P', 'members3', 'cpm3', 'pL']].agg(['mean', 'size']).round(2))
    g2 = g[g.pE > 1]
    g2 = g2.assign(bL=pd.cut(g2.pL, [-1, 0.05, 0.3, 1, 2, 30]))
    print(' with early punishment >1/member:')
    print(g2.groupby('bL')[['P', 'members3', 'cpm3']].agg(['mean', 'size']).round(2))
m.to_csv('/tmp/v1m.csv', index=False)
