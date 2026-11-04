"""v3 sweep: group-level panel of rounds 0-23 (my group = agent_group 0), saved to a cache parquet;
then the first decision's member change against the relative common good of the two groups."""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
pd.set_option('display.width', 250)
B = 'plots/simulation/policy_finder/pf-20261104-7-v3_sweep/'
CACHE = '/tmp/v3_groups.parquet'
if not os.path.exists(CACHE):
    out = []
    for part in ['p1of2', 'p2of2']:
        f = pq.ParquetFile(f'{B}per_round_{part}.parquet')
        for i in range(f.num_row_groups):
            d = f.read_row_group(i, columns=['run', 'episode', 'round_number', 'punishment', 'contribution', 'agent_group', 'common_good']).to_pandas()
            d['run'] = d.run.astype(str).str.extract(r'_s(\d+)_vs')[0].astype(int)
            g = d.groupby(['run', 'episode', 'round_number', 'agent_group']).agg(
                n=('contribution', 'size'), c=('contribution', 'mean'), p=('punishment', 'mean'), cg=('common_good', 'first')).reset_index()
            out.append(g)
    G = pd.concat(out).groupby(['run', 'episode', 'round_number', 'agent_group']).agg(
        n=('n', 'sum'), c=('c', 'mean'), p=('p', 'mean'), cg=('cg', 'first')).reset_index()
    G.to_parquet(CACHE)
G = pd.read_parquet(CACHE)
print(G.head()); print(G.groupby('agent_group').n.mean())

# wide panel, empty groups filled
W = G.pivot_table(index=['run', 'episode', 'round_number'], columns='agent_group', values=['n', 'c', 'p', 'cg']).fillna(0)
W.columns = [f'{a}{b}' for a, b in W.columns]
W = W.reindex(pd.MultiIndex.from_product([sorted(G.run.unique()), sorted(G.episode.unique()), range(24)],
                                         names=['run', 'episode', 'round_number']), fill_value=0).reset_index()
W['pool0'] = W.cg0 * W.n0  # approx (cg is per contributing member)
W.to_parquet('/tmp/v3_wide.parquet')
ep = W.groupby(['run', 'episode']).pool0.mean().rename('score').reset_index()
r = {k: W[W.round_number == k].set_index(['run', 'episode']) for k in range(24)}
E = pd.DataFrame({'n4': r[4].n0, 'n3': r[3].n0, 'p03': W[W.round_number < 4].groupby(['run', 'episode']).p0.mean(),
                  'p3': r[3].p0, 'dcg3': r[3].cg0 - r[3].cg1,
                  'dcg02': W[W.round_number < 3].assign(d=lambda x: x.cg0 - x.cg1).groupby(['run', 'episode']).d.mean(),
                  'c3': r[3].c0, 'c0': r[0].c0}).join(ep.set_index(['run', 'episode']))
E['dn'] = E.n4 - E.n3
print('first decision: member change by round-3 cg gap (mine - other)')
E['gb'] = pd.cut(E.dcg3, [-60, -15, -10, -5, -2, 0, 2, 5, 10, 15, 60])
print(E.groupby('gb', observed=True).agg(dn=('dn', 'mean'), n4=('n4', 'mean'), empty=('n4', lambda x: (x == 0).mean()), N=('dn', 'size'), score=('score', 'mean')).round(2))
X = np.column_stack([np.ones(len(E)), E.dcg3, E.dcg02, E.p3, E.p03, E.n3, E.c3])
b, *_ = np.linalg.lstsq(X, E.dn.values, rcond=None)
print('OLS dn on 1, dcg3, dcg02, p3, p0-3 mean, n3, c3', b.round(3))
E.drop(columns='gb').to_parquet('/tmp/v3_ep.parquet')
