# v7: pool (opponent-demeaned) by round-0 breadth x depth, from /tmp/v7_breadth.csv (v7_s4_breadth.py)
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
a = pd.read_csv('/tmp/v7_breadth.csv')
a['pool_d'] = a.pool - a.groupby('opp').pool.transform('mean')
a['mem_d'] = a.members - a.groupby('opp').members.transform('mean')
a['depth0'] = a.p_r0 / a.hit_r0.clip(lower=1e-3)
a['depth1'] = a.p_r1 / a.hit_r1.clip(lower=1e-3)
a['hb'] = pd.cut(a.hit_r0, [0, .25, .35, .45, .55, .65, 1])
a['db'] = pd.cut(a.depth0, [0, 5, 10, 15, 20, 25, 30])
for val in ['pool_d', 'mem_d', 'c_mid', 'c_end']:
    print(val); print(a.pivot_table(index='hb', columns='db', values=val, aggfunc='mean', observed=True).round(2))
print(a.pivot_table(index='hb', columns='db', values='pool_d', aggfunc='size', observed=True))
# among deep (depth0>20) openings: round 1 handling
s = a[(a.depth0 > 20)]
s['h1b'] = pd.cut(s.hit_r1, [-.01, .1, .2, .3, .4, .5, 1]); s['d1b'] = pd.cut(s.depth1, [-0.1, 5, 10, 20, 30])
print(s.pivot_table(index='h1b', columns='d1b', values='pool_d', aggfunc='mean', observed=True).round(2))
print(s.pivot_table(index='h1b', columns='d1b', values='pool_d', aggfunc='size', observed=True))
# by version: best (point-mean over opps) of deep narrow
a['grp'] = np.where((a.hit_r0 < .45) & (a.depth0 > 20), 'narrow_deep', 'other')
print(a.groupby(['v', 'grp']).pool_d.agg(['mean', 'size']).round(2))
