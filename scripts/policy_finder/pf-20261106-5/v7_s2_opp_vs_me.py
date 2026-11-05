# v7: in the finals, my group vs the opponent's group per round: members, mean c, mean p, pool
import sys, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
v = sys.argv[1] if len(sys.argv) > 1 else 'pf-20261106-5-v6'
cols = ['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run', 'common_good']
pt = sys.argv[2] if len(sys.argv) > 2 else '_gp_'
import pyarrow.parquet as pq, pyarrow.compute as pc
t = pq.read_table(base + v + '_sweep/final/per_round.parquet', columns=cols)
t = t.filter(pc.match_substring(t['run'], pt))
d = t.to_pandas()
d['run'] = d.run.astype('category')
d['pair'] = d.run.str.split('managed by ').str[1]
d['me_g0'] = d.pair.str.startswith('pf-')
d['opp'] = d.pair.str.extract(r'(opponent_\d)')[0]
# group_id: assume 0/1 ; which group is mine
d['mine'] = np.where(d.me_g0, d.group_id == 0, d.group_id == 1)
g = d.groupby(['opp', 'mine', 'run', 'episode', 'round_number'], observed=True)
a = g.agg(n=('contribution', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean'), pool=('common_good', 'first')).reset_index()
a['pool2'] = 1.6 * a.c * a.n - a.p * a.n
print(a[['pool', 'pool2']].describe())
# full game summary with empty group-rounds as 0? show mean pool per round (episodes x rounds)
s = a.groupby(['opp', 'mine'])[['n', 'c', 'p', 'pool2']].mean().round(2)
print(s)
a['blk'] = a.round_number // 4
print(a.pivot_table(index=['opp', 'blk'], columns='mine', values=['n', 'c', 'p'], aggfunc='mean').round(2))
