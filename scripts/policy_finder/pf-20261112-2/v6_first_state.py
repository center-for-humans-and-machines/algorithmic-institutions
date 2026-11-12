"""v6: v5_first_state on the per-game baseline pairs of v6_gamebase (v4 and v5 sweeps only). v5: state of the race at each paired game's FIRST fine (identical in both arms): per-capita
pool of my group and ah's (that round and averaged over the block so far, before my fine),
both sizes, the target's contribution and shortfall, how many I fine that round.
Writes /tmp/v6_first.parquet (pairs joined with the state)."""
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq
P = pd.read_parquet('/tmp/v6_H.parquet')
P = P[(P.spend > 0) & P.lead0.notna() & P.ver.isin(['pf-20261112-2-v4', 'pf-20261112-2-v5'])]
want = P[['ver', 'idx']].drop_duplicates()
out = []
for v in want.ver.unique():
    ids = set(want[want.ver == v].idx)
    for f in sorted(glob.glob(f'plots/simulation/policy_finder/{v}_sweep/per_round*.parquet')):
        runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
        runs = [r for r in runs_all if int(r.split('_s')[-1].split('_vs')[0]) in ids]
        for k0 in range(0, len(runs), 20):
            d = pq.read_table(f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run'],
                              filters=[('run', 'in', runs[k0:k0 + 20])]).to_pandas()
            r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
            first = r[0].str.startswith('pf')
            pt = np.where(first, r[0], r[1])
            d['idx'] = pd.Series(pt).str.extract(r'_s(\d+)$')[0].astype(int).values
            d['seat'] = np.where(first, 0, 1)
            d['mine'] = (d.group_id.values == d.seat.values).astype(int)
            d['pme'] = d.punishment * d.mine
            g = ['idx', 'seat', 'episode']
            tf = d[d.pme > 0].groupby(g).round_number.min().rename('tf')
            d = d.join(tf, on=g, how='inner')
            d = d[(d.round_number <= d.tf) & (d.round_number >= (d.tf // 4) * 4)]
            d['pah'] = d.punishment * (1 - d.mine)
            key = g + ['round_number', 'mine']
            a = d.groupby(key).agg(n=('contribution', 'size'), nv=('contribution_valid', 'sum'), cs=('contribution', 'sum'), ps=('pah', 'sum')).reset_index()
            a['pc'] = (1.6 * a.cs - a.ps) / a.nv.clip(lower=1)
            w = a.pivot_table(index=g + ['round_number'], columns='mine', values=['n', 'pc', 'cs', 'nv']).fillna(0)
            w.columns = [f'{x}_{"me" if m else "ah"}' for x, m in w.columns]
            w = w.reset_index().join(tf, on=g)
            w['pcgap'] = w.pc_me - w.pc_ah
            blk = w.groupby(g).pcgap.mean().rename('pcgap_blk')
            now = w[w.round_number == w.tf].set_index(g)
            e = d[(d.round_number == d.tf) & (d.pme > 0)].groupby(g).agg(n_fined=('pme', 'size'), c_tgt=('contribution', 'mean'), fine1=('pme', 'mean'))
            s = now.join(blk).join(e).reset_index()
            s['ver'] = v
            out.append(s)
    print(v, flush=True)
S = pd.concat(out)
X = P.merge(S, on=['ver', 'idx', 'seat', 'episode'])
X['oth_mean'] = (X.cs_me - X.c_tgt * X.n_fined) / (X.nv_me - X.n_fined).clip(lower=1)
X['short'] = X.oth_mean - X.c_tgt
X.to_parquet('/tmp/v6_first.parquet')
print(len(X))
