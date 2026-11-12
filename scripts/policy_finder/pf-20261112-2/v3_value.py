"""v3: the value to my pool of fining one member, by the round's phase in its block.
In v1's sweep (fines in every round up to t_stop, size and target varying with the random
design point), for each valid member i of my group at round t, outcomes at t+k (k=1..8):
  S_k = c_i(t+k) if i is still in my group else 0   (the target's own future pool input)
  O_k = sum of c(t+k) over the round-t co-members still in my group  (spillover)
  N_k = my group's size at t+k
Saved to /tmp/v3_value.parquet for v3_value_analyse.py."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, pyarrow.parquet as pq
base = 'plots/simulation/policy_finder/pf-20261112-2_sweep/'
H = 8
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
        rng = np.random.default_rng(k0 + len(recs))
        for tt in range(1, 23):
            e, i = np.nonzero(mine[:, :, tt] & V[:, :, tt])
            keep = rng.random(len(e)) < 0.3
            e, i = e[keep], i[keep]
            oth = mine[e, :, tt].copy(); oth[np.arange(len(e)), i] = False
            no = np.maximum(oth.sum(1), 1)
            rec = {'t': tt, 'c': C[e, i, tt], 'cp': C[e, i, tt - 1], 'p': P[e, i, tt], 'om': (C[e, :, tt] * oth).sum(1) / no,
                   'op': (P[e, :, tt] * oth).sum(1), 'n': oth.sum(1) + 1, 'pprev': P[e, i, tt - 1]}
            for k in range(1, H + 1):
                if tt + k > 23:
                    rec[f'S{k}'] = np.nan; rec[f'O{k}'] = np.nan; rec[f'N{k}'] = np.nan; continue
                rec[f'S{k}'] = np.where(mine[e, i, tt + k], C[e, i, tt + k], 0)
                rec[f'O{k}'] = (C[e, :, tt + k] * oth * mine[e, :, tt + k]).sum(1)
                rec[f'N{k}'] = mine[e, :, tt + k].sum(1)
            recs.append(pd.DataFrame(rec).astype('float32'))
x = pd.concat(recs)
x.to_parquet('/tmp/v3_value.parquet')
print(len(x))
