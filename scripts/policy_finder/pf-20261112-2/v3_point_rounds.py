"""v3: per design point, seat and round, mean over episodes of my group's and ah's
pool (1.6*sum c - sum p, empty group 0), size, contributions and fines.
Writes /tmp/v3_pr_<version>.parquet."""
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq
v = sys.argv[1]
base = f'plots/simulation/policy_finder/{v}_sweep/'
out = []
for f in sorted(glob.glob(base + 'per_round*.parquet')):
    runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
    for k0 in range(0, len(runs_all), 32):
        d = pq.read_table(f, columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'], filters=[('run', 'in', runs_all[k0:k0 + 32])]).to_pandas()
        r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
        first = r[0].str.startswith('pf')
        d['pt'] = np.where(first, r[0], r[1]); d['seat'] = np.where(first, 0, 1)
        d['mine'] = (d.group_id.values == d.seat.values)
        g = d.groupby(['pt', 'seat', 'episode', 'round_number', 'mine']).agg(n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum'))
        g = g.unstack('mine', fill_value=0)
        g.columns = [f'{a}_{"me" if m else "ah"}' for a, m in g.columns]
        g = g.reset_index()
        for s in ['me', 'ah']:
            g[f'pool_{s}'] = 1.6 * g[f'c_{s}'] - g[f'p_{s}']
        out.append(g.groupby(['pt', 'seat', 'round_number']).agg(['mean']).droplevel(1, axis=1).drop(columns='episode').reset_index().assign(ne=g.groupby(['pt', 'seat', 'round_number']).size().values))
x = pd.concat(out)
x.to_parquet(f'/tmp/v3_pr_{v}.parquet')
print(x.groupby('round_number')[['n_me', 'n_ah', 'pool_me', 'pool_ah', 'p_me', 'p_ah']].mean().round(2))
