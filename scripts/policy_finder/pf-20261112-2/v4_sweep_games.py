"""v4: per game (version, point index, seat, episode) of each sweep: lead (mean over rounds of
my pool minus ah's), my spend, first round I fine, my size at the end. Writes /tmp/v4_games.parquet."""
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq
vs = ['pf-20261112-2', 'pf-20261112-2-v2', 'pf-20261112-2-v3']
out = []
for v in vs:
    for part, f in enumerate(sorted(glob.glob(f'plots/simulation/policy_finder/{v}_sweep/per_round*.parquet'))):
        runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
        for k0 in range(0, len(runs_all), 40):
            d = pq.read_table(f, columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'],
                              filters=[('run', 'in', runs_all[k0:k0 + 40])]).to_pandas()
            r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
            first = r[0].str.startswith('pf')
            pt = np.where(first, r[0], r[1])
            d['idx'] = pd.Series(pt).str.extract(r'_s(\d+)$')[0].astype(int).values
            d['seat'] = np.where(first, 0, 1)
            mine = (d.group_id.values == d.seat.values)
            d['lead'] = (1.6 * d.contribution - d.punishment) * np.where(mine, 1, -1) / 24
            d['pme'] = d.punishment * mine
            d['mine'] = mine.astype(int)
            g = d.groupby(['idx', 'seat', 'episode']).agg(lead=('lead', 'sum'), spend=('pme', 'sum'))
            tf = d[d.pme > 0].groupby(['idx', 'seat', 'episode']).round_number.min().rename('t_first')
            ne = d[d.round_number == 23].groupby(['idx', 'seat', 'episode']).mine.sum().rename('n_end')
            g = g.join(tf).join(ne).reset_index()
            g['ver'] = v; g['part'] = part
            out.append(g)
    print(v, flush=True)
G = pd.concat(out)
G.to_parquet('/tmp/v4_games.parquet')
print(G.groupby('ver').size())
