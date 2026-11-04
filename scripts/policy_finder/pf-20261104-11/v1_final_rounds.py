"""v1 final: per point, by round: own group size, pool, punishment, contribution; ah group too."""
import sys, pandas as pd, numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'plots/simulation/policy_finder/pf-20261104-11_sweep/final/per_round.parquet'
d = pd.read_parquet(path)
d['pt'] = d.run.str.extract(r'managed by (\S+_vs_\S+)$')[0]
print(d.groupby(['pt','group_id']).size())
# which group_id is the rule's group? check punishment pattern
g = d.groupby(['pt', 'episode', 'round_number', 'group_id']).agg(
    n=('contribution', 'size'), csum=('contribution', 'sum'), psum=('punishment', 'sum'),
    cbar=('contribution','mean')).reset_index()
g['pool'] = 1.6 * g.csum - g.psum
for pt, gg in g.groupby('pt'):
    ne = gg.episode.nunique()
    print('\n==', pt, ne)
    piv = gg.groupby(['group_id', 'round_number'])[['n', 'pool', 'psum', 'cbar']].sum().div(ne)
    piv['cbar'] = gg.groupby(['group_id', 'round_number']).cbar.mean()
    print(piv.unstack('group_id').round(2).to_string())
