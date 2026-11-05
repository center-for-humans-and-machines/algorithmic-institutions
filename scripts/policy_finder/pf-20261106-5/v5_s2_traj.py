# v5: per-round trajectories of the finals' peaks (v3, v4) per opponent, my group vs the other
import sys, re, numpy as np, pandas as pd, pyarrow.parquet as pq
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ver = sys.argv[1] if len(sys.argv) > 1 else 'pf-20261106-5-v4'
point = sys.argv[2] if len(sys.argv) > 2 else '_gp'
d = pq.read_table(base + ver + '_sweep/final/per_round.parquet').to_pandas()
d = d[d.run.str.contains(re.escape(ver + point) + r'(_vs|$)')]
d['seat'] = np.where(d.run.str.contains('managed by pf-'), 0, 1)
d['opp'] = d.run.str.extract(r'(opponent_\d)')[0]
d['mine'] = d.group_id == d.seat
d['valid'] = d.contribution_valid.astype(float)
g = d.groupby(['opp', 'seat', 'episode', 'round_number', 'mine']).agg(n=('valid', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean'),
                                                                    pool=('common_good', 'first')).reset_index()
r = g.groupby(['opp', 'round_number', 'mine'])[['n', 'c', 'p', 'pool']].mean().unstack('mine')
for o in sorted(d.opp.unique()):
    x = r.loc[o]; x.columns = [f'{a}_{"me" if b else "oth"}' for a, b in x.columns]
    print(o); print(x.round(2).T.to_string())
# overall
tot = g[g.mine].groupby(['opp', 'episode', 'seat']).pool.sum().groupby('opp').mean() / 24
print('mean pool per round (empty rounds missing -> counts only present)', tot.round(2).to_dict())
