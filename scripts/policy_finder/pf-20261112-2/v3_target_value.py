"""v3: design-level value of spend by the target's contribution and by period.
Per v1 design point: my spend (sum of fines per game) on members giving 0, 1-4, 5-9, 10+,
split into rounds 0-3 and 4-19 (endgame shown in v3_endgame to be pure cost);
regress the lead (mean over rounds of my pool - ah's) on these spends / 24
(coefficient -1 = pure cost, no return)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, json, numpy.linalg as la, pyarrow.parquet as pq
base = 'plots/simulation/policy_finder/pf-20261112-2_sweep/'
rows = []
for f in sorted(glob.glob(base + 'per_round_p*.parquet')):
    runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
    for k0 in range(0, len(runs_all), 16):
        d = pq.read_table(f, columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'], filters=[('run', 'in', runs_all[k0:k0 + 16])]).to_pandas()
        r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
        first = r[0].str.startswith('pf')
        d['pt'] = np.where(first, r[0], r[1]); d['seat'] = np.where(first, 0, 1)
        d = d[d.group_id.values == d.seat.values]
        d['cb'] = pd.cut(d.contribution, [-1, 0, 4, 9, 20], labels=['c0', 'c1_4', 'c5_9', 'c10p']).astype(str)
        d['per'] = np.select([d.round_number < 4, d.round_number < 20], ['r0_3', 'r4_19'], 'r20_23')
        ne = d.groupby(['pt', 'seat']).episode.nunique()
        s = d.groupby(['pt', 'seat', 'per', 'cb']).punishment.sum().unstack(['per', 'cb'], fill_value=0)
        s = s.div(ne, axis=0)
        rows.append(s)
s = pd.concat(rows).fillna(0).groupby('pt').mean()
s.columns = [f'{a}_{b}' for a, b in s.columns]
x = pd.read_parquet('/tmp/v3_pr_pf-20261112-2.parquet')
x['lead'] = x.pool_me - x.pool_ah
lead = x.groupby('pt').lead.mean()
s = s.join(lead)
s.to_csv('scripts/policy_finder/pf-20261112-2/v3_target_value.csv')
cols = [c for c in s.columns if c != 'lead' and not c.startswith('r20') and s[c].mean() > 0.5]
print(s[cols].mean().round(1))
X = np.column_stack([np.ones(len(s))] + [s[c] / 24 for c in cols] + [s[[c for c in s.columns if c.startswith('r20')]].sum(1) / 24])
b, *_ = la.lstsq(X, s.lead.values, rcond=None); r = s.lead.values - X @ b
se = np.sqrt(np.diag(la.inv(X.T @ X)) * r.var() * len(r) / (len(r) - X.shape[1]))
for n, bb, ss in zip(['const'] + cols + ['endgame'], b, se): print(f'{n:12s} {bb:7.2f} +- {ss:5.2f}')
