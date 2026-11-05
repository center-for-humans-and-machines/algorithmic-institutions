# v7: distribution of contributions in my group vs theirs by block (final gp), and individual persistence of low contributors
import sys, pandas as pd, numpy as np
import pyarrow.parquet as pq, pyarrow.compute as pc
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
v = sys.argv[1] if len(sys.argv) > 1 else 'pf-20261106-5-v6'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run', 'contribution_valid']
t = pq.read_table(base + v + '_sweep/final/per_round.parquet', columns=cols)
t = t.filter(pc.match_substring(t['run'], '_gp_'))
d = t.to_pandas()
d['me_g0'] = d.run.str.contains('managed by pf-')
d['opp'] = d.run.str.extract(r'(opponent_\d)')[0]
d['mine'] = np.where(d.me_g0, d.group_id == 0, d.group_id == 1)
d['blk'] = d.round_number // 4
d['cb'] = pd.cut(d.contribution, [-1, 0, 5, 9, 12, 15, 19, 20])
print(pd.crosstab([d.mine, d.blk], d.cb, normalize='index').round(3))
print('valid share', d.groupby('mine').contribution_valid.mean())
# persistence: player's mean c in rounds 0-3 vs mean c in rounds 12-23 when in my group
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
key = ['run', 'episode', 'participant_code']
e = d[d.round_number < 4].groupby(key, observed=True).agg(c_early=('contribution', 'mean'), mine0=('mine', 'first'), pun0=('punishment', 'sum'))
l = d[d.round_number >= 12].groupby(key, observed=True).agg(c_late=('contribution', 'mean'), mine_late=('mine', 'mean'))
x = e.join(l)
x['ceb'] = pd.cut(x.c_early, [-1, 3, 6, 9, 12, 15, 20])
print(x.groupby(['mine0', 'ceb'], observed=True)[['c_late', 'mine_late', 'pun0']].agg(['mean', 'size']).round(2))
