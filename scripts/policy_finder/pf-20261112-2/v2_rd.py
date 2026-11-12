"""v2: regression discontinuity at v1's threshold. In v1 a member is fined iff c < thr
(and t <= t_stop, fine = floor(base + slope*(thr-c)) >= 1). Compare members at
c = c* (highest fined integer) and c*+1 (lowest unfined): their future value to my pool
V = sum_{s>t} (1.6 c_s - p_s) while in my group, minus the fine now; their next-round
contribution; staying in my group at the next arrival. Also the others' next-round
contributions (spillover) and others' future value. Slope correction from c*+1 vs c*+2."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, json, pyarrow.parquet as pq
base = 'plots/simulation/policy_finder/pf-20261112-2_sweep/'
pts = {p['name']: p['params'] for p in json.load(open(base + 'sweep.json'))['points']}
recs = []
for f in sorted(glob.glob(base + 'per_round_p*.parquet')):
    runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
    for k0 in range(0, len(runs_all), 32):
        t = pq.read_table(f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run'], read_dictionary=['run', 'participant_code'], filters=[('run', 'in', runs_all[k0:k0 + 32])])
        d = t.to_pandas()
        cats = pd.Series(d.run.cat.categories)
        r = cats.str.extract(r'by (\S+)_vs_(\S+)$')
        first = r[0].str.startswith('pf').values
        name = np.where(first, r[0], r[1])
        code = d.run.cat.codes.values.astype(np.int64)
        pcat = pd.Series(d.participant_code.cat.categories).str.split('_').str[0].astype(int).values
        d['pid'] = pcat[d.participant_code.cat.codes.values.astype(np.int64)]
        d['code'] = code
        d['ep'] = code * 10000 + d.episode.values
        d = d.drop(columns=['participant_code', 'run']).sort_values(['ep', 'pid', 'round_number'])
        E = d.ep.nunique()
        A = lambda col: d[col].values.reshape(E, 8, 24)
        P, C, V, G, CODE = A('punishment'), A('contribution'), A('contribution_valid'), A('group_id'), A('code')[:, 0, 0]
        MY = np.where(first[CODE], 0, 1)
        mine = G == MY[:, None, None]
        net = np.where(mine, 1.6 * C - P, 0.0)
        fut = np.cumsum(net[:, :, ::-1], 2)[:, :, ::-1]  # fut[..., s] = sum_{u>=s}
        nmine = mine.sum(1)
        for ci in np.unique(CODE):
            pr = pts[name[ci]]
            thr, b, sl, ts = pr['thr'], pr['base'], pr['slope'], pr['t_stop']
            cstar = int(np.ceil(thr)) - 1
            fine = int(np.floor(b + sl * (thr - cstar)))
            if fine < 1 or cstar + 2 > 20 or cstar < 0:
                continue
            es = np.nonzero(CODE == ci)[0]
            for tt in range(0, min(ts, 22) + 1):
                for off in [0, 1, 2]:
                    cc = cstar + off
                    e, i = np.nonzero(mine[es, :, tt] & V[es, :, tt] & (C[es, :, tt] == cc))
                    e = es[e]
                    if len(e) == 0: continue
                    nxt = min(((tt // 4) + 1) * 4, 23)
                    oth = mine[e, :, tt].copy(); oth[np.arange(len(e)), i] = False
                    no = oth.sum(1)
                    oc1 = np.where(oth & mine[e, :, tt + 1], C[e, :, tt + 1], 0).sum(1) / np.maximum((oth & mine[e, :, tt + 1]).sum(1), 1)
                    oc0 = np.where(oth, C[e, :, tt], 0).sum(1) / np.maximum(no, 1)
                    recs.append(pd.DataFrame({'pt': ci + 1000 * (f.endswith('p2of2.parquet')), 'fine': fine, 'cstar': cstar, 't': tt, 'off': off,
                        'p_now': P[e, i, tt], 'V': fut[e, i, tt + 1], 'c1': np.where(mine[e, i, tt + 1], C[e, i, tt + 1], np.nan),
                        'stay': mine[e, i, nxt].astype(float), 'oV': (fut[e, :, tt + 1] * oth).sum(1), 'odc': oc1 - oc0,
                        'nfut': nmine[e, tt + 1:].sum(1)}))
x = pd.concat(recs)
x.to_parquet('/tmp/v2_rd.parquet')
print(len(x))
