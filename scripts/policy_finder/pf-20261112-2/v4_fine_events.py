"""v4: every fine my manager dealt in the fining points that have a never-fining partner at
the same index (the paired set of v4_sweep_crn.py): game, round, amount, the target's
contribution, my and ah's group sizes and mean contributions that round. /tmp/v4_events.parquet"""
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq
P = pd.read_parquet('/tmp/v4_pairs.parquet')
want = P[['ver', 'idx']].drop_duplicates()
out = []
for v in want.ver.unique():
    ids = set(want[want.ver == v].idx)
    for f in sorted(glob.glob(f'plots/simulation/policy_finder/{v}_sweep/per_round*.parquet')):
        runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
        runs = [r for r in runs_all if int(r.split('_s')[-1].split('_vs')[0]) in ids]
        for k0 in range(0, len(runs), 30):
            d = pq.read_table(f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run'],
                              filters=[('run', 'in', runs[k0:k0 + 30])]).to_pandas()
            r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
            first = r[0].str.startswith('pf')
            pt = np.where(first, r[0], r[1])
            d['idx'] = pd.Series(pt).str.extract(r'_s(\d+)$')[0].astype(int).values
            d['seat'] = np.where(first, 0, 1)
            d['mine'] = (d.group_id.values == d.seat.values).astype(int)
            key = ['idx', 'seat', 'episode', 'round_number']
            grp = d.groupby(key + ['mine']).agg(n=('contribution', 'size'), cs=('contribution', 'sum')).unstack('mine', fill_value=0)
            grp.columns = [f'{a}_{"me" if m else "ah"}' for a, m in grp.columns]
            e = d[(d.mine == 1) & (d.punishment > 0)][key + ['participant_code', 'punishment', 'contribution']]
            e = e.merge(grp.reset_index(), on=key)
            e['ver'] = v
            out.append(e)
    print(v, flush=True)
E = pd.concat(out)
E.to_parquet('/tmp/v4_events.parquet')
print(len(E))
