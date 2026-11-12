"""v2: contribution response to my rule's fines in v1's sweep. Conditional on c_t, whether a
member is fined varies across design points (thr, base, slope, t_stop are random), so
regress c_{t+k} (same group) on the fine p_t with c_t dummies, c_{t-1}, the others' mean,
and round dummies; by contribution band. Also the same for others' response (spillover):
others' mean c_{t+1} on the total fines in the group."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq
base = 'plots/simulation/policy_finder/pf-20261112-2_sweep/'
recs = []
for f in sorted(glob.glob(base + 'per_round_p*.parquet')):
    runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
    for k0 in range(0, len(runs_all), 32):
        t = pq.read_table(f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run'], read_dictionary=['run', 'participant_code'], filters=[('run', 'in', runs_all[k0:k0 + 32])])
        d = t.to_pandas()
        cats = pd.Series(d.run.cat.categories)
        first = cats.str.extract(r'by (\S+)_vs_')[0].str.startswith('pf').values
        code = d.run.cat.codes.values.astype(np.int64)
        pcat = pd.Series(d.participant_code.cat.categories).str.split('_').str[0].astype(int).values
        d['pid'] = pcat[d.participant_code.cat.codes.values.astype(np.int64)]
        d['code'] = code; d['ep'] = code * 10000 + d.episode.values
        d = d.drop(columns=['participant_code', 'run']).sort_values(['ep', 'pid', 'round_number'])
        E = d.ep.nunique()
        A = lambda col: d[col].values.reshape(E, 8, 24)
        P, C, V, G, CODE = A('punishment'), A('contribution'), A('contribution_valid'), A('group_id'), A('code')[:, 0, 0]
        MY = np.where(first[CODE], 0, 1)
        mine = G == MY[:, None, None]
        rng = np.random.default_rng(k0)
        for tt in range(1, 21):
            e, i = np.nonzero(mine[:, :, tt] & V[:, :, tt])
            keep = rng.random(len(e)) < 0.35
            e, i = e[keep], i[keep]
            oth = mine[e, :, tt].copy(); oth[np.arange(len(e)), i] = False
            no = np.maximum(oth.sum(1), 1)
            rec = {'t': tt, 'c': C[e, i, tt], 'cp': C[e, i, tt - 1], 'p': P[e, i, tt], 'om': (C[e, :, tt] * oth).sum(1) / no,
                   'op': (P[e, :, tt] * oth).sum(1), 'n': oth.sum(1) + 1, 'pprev': P[e, i, tt - 1]}
            for k in [1, 2, 3]:
                same = mine[e, i, tt + k]
                rec[f'c{k}'] = np.where(same, C[e, i, tt + k], np.nan)
            recs.append(pd.DataFrame(rec))
x = pd.concat(recs)
x.to_parquet('/tmp/v2_resp.parquet')
import numpy.linalg as la
for lo, hi in [(0, 0), (1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]:
    for k in [1, 2, 3]:
        y = x[(x.c >= lo) & (x.c <= hi)].dropna(subset=[f'c{k}'])
        X = np.column_stack([np.ones(len(y)), y.p, (y.p > 0), y.op, y.cp, y.om, y.n, y.pprev] + [(y.c == v).values for v in range(lo + 1, hi + 1)] + [(y.t == v).values for v in range(2, 21)])
        b = la.lstsq(X, y[f'c{k}'].values, rcond=None)[0]
        print(f'c {lo}-{hi} lag {k}: n={len(y)} fined share={(y.p>0).mean():.2f} per-point={b[1]:.3f} any-fine={b[2]:.3f} others-fines={b[3]:.3f}')
