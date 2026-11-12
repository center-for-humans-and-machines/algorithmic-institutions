"""v7: lean rewrite of v6_sweep_games (that one ran out of memory on 15 GB). Reads each sweep file
row group by row group with `run` dictionary-encoded and aggregates per game (version, point index,
seat, episode): lead (mean over rounds of my pool minus ah's), my spend, first round I fine, my size
at round 23, and my spend split by block. Usage: v7_games_lean.py <version> ... ; writes
/tmp/v7_games_<version>.parquet per version."""
import sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq

for v in sys.argv[1:]:
    parts = []
    for f in sorted(glob.glob(f'plots/simulation/policy_finder/{v}_sweep/per_round*.parquet')):
        pf = pq.ParquetFile(f, read_dictionary=['run'])
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'])
            run = t.column('run').combine_chunks()
            names = run.dictionary.to_pylist()
            codes = run.indices.to_numpy()
            idx_n = np.array([int((n.split('by ')[1].split('_vs_')[0] if n.split('by ')[1].startswith('pf') else n.split('_vs_')[1]).split('_s')[-1]) for n in names])
            seat_n = np.array([0 if n.split('by ')[1].startswith('pf') else 1 for n in names])
            idx = idx_n[codes]; seat = seat_n[codes]
            ep = t.column('episode').to_numpy(); rn = t.column('round_number').to_numpy()
            pu = t.column('punishment').to_numpy().astype(float); co = t.column('contribution').to_numpy().astype(float)
            gid = t.column('group_id').to_numpy()
            mine = (gid == seat)
            df = pd.DataFrame({'idx': idx, 'seat': seat, 'episode': ep, 'lead': (1.6 * co - pu) * np.where(mine, 1, -1) / 24,
                               'spend': pu * mine, 'tf': np.where(mine & (pu > 0), rn, 99),
                               'n_end': (mine & (rn == 23)).astype(float)})
            for b in range(6):
                df[f'sp{b}'] = df.spend * (rn // 4 == b)
            parts.append(df.groupby(['idx', 'seat', 'episode']).agg(
                lead=('lead', 'sum'), spend=('spend', 'sum'), tf=('tf', 'min'), n_end=('n_end', 'sum'),
                **{f'sp{b}': (f'sp{b}', 'sum') for b in range(6)}))
            del t, df
    P = pd.concat(parts)
    g = P.groupby(level=[0, 1, 2]).agg({'lead': 'sum', 'spend': 'sum', 'tf': 'min', 'n_end': 'sum', **{f'sp{b}': 'sum' for b in range(6)}}).reset_index()
    g['ver'] = v
    g.to_parquet(f'/tmp/v7_games_{v}.parquet')
    print(v, len(g), g.groupby('idx').size().describe().to_dict(), flush=True)
