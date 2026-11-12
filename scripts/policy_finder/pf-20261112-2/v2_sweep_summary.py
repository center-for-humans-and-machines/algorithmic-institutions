"""v2: per design point of the v1 sweep, my group's spend, members, per-capita pool, vs score."""
import pandas as pd, numpy as np, json, glob, pyarrow.parquet as pq, pyarrow.compute as pc
base = 'plots/simulation/policy_finder/pf-20261112-2_sweep/'
rows = []
for f in sorted(glob.glob(base + 'per_round_p*.parquet')):
  pf = pq.ParquetFile(f)
  for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'])
    t = t.set_column(5, 'run', pc.dictionary_encode(t.column('run')))
    d = t.to_pandas()
    cats = pd.Series(d.run.cat.categories)
    r = cats.str.extract(r'by (\S+)_vs_(\S+)$')
    first = r[0].str.startswith('pf').values
    code = d.run.cat.codes.values
    mine_g = np.where(first[code], 0, 1)
    d['mine'] = (d.group_id.values == mine_g)
    d['pt'] = np.where(first, r[0], r[1])[code]; d['seat'] = first[code]
    d = d.drop(columns='run')
    g = d.groupby(['pt', 'seat', 'mine', 'episode', 'round_number']).agg(n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum')).reset_index()
    g['pool'] = 1.6 * g.c - g.p
    a = g.groupby(['pt', 'mine']).agg(n=('n', 'mean'), pool=('pool', 'mean'), c=('c', 'sum'), p=('p', 'sum'), nn=('n', 'sum')).reset_index()
    a['cpc'] = a.c / a.nn; a['ppc'] = a.p / a.nn
    rows.append(a)
a = pd.concat(rows).groupby(['pt', 'mine']).mean().reset_index()
w = a.pivot(index='pt', columns='mine', values=['n', 'pool', 'cpc', 'ppc'])
w.columns = [f'{x}_{"me" if m else "ah"}' for x, m in w.columns]
w['score'] = w.pool_me - w.pool_ah
w = w.sort_values('ppc_me')
w['spb'] = pd.cut(w.ppc_me, [-.01, .05, .2, .4, .7, 1, 1.5, 2.5, 10])
print(w.groupby('spb', observed=True).agg(k=('score', 'size'), score=('score', 'mean'), n_me=('n_me', 'mean'), cpc_me=('cpc_me', 'mean'), cpc_ah=('cpc_ah', 'mean'), ppc_ah=('ppc_ah', 'mean')).round(2))
w.to_csv('scripts/policy_finder/pf-20261112-2/v2_sweep_summary.csv')
