"""v3: v3_target_value on v2's sweep, where the fining window w (ending at the decision
round) separates decision-round fines (phase 3) from earlier-in-block fines (phases 0-2).
Spend by period (rounds 0-3 / 4-19), target band (0 / 1-9 / 10+) and phase (0-2 / 3)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, glob, numpy.linalg as la, pyarrow.parquet as pq
base = 'plots/simulation/policy_finder/pf-20261112-2-v2_sweep/'
rows = []
for f in sorted(glob.glob(base + 'per_round_p*.parquet')):
    runs_all = sorted(set(pq.read_table(f, columns=['run']).column('run').unique().to_pylist()))
    for k0 in range(0, len(runs_all), 16):
        d = pq.read_table(f, columns=['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run'], filters=[('run', 'in', runs_all[k0:k0 + 16])]).to_pandas()
        r = d.run.str.extract(r'by (\S+)_vs_(\S+)$')
        first = r[0].str.startswith('pf')
        d['pt'] = np.where(first, r[0], r[1]); d['seat'] = np.where(first, 0, 1)
        d = d[d.group_id.values == d.seat.values]
        d['cb'] = pd.cut(d.contribution, [-1, 0, 9, 20], labels=['c0', 'c1_9', 'c10p']).astype(str)
        d['per'] = np.select([d.round_number < 4, d.round_number < 20], ['early', 'mid'], 'end')
        d['ph'] = np.where(d.round_number % 4 == 3, 'dec', 'pre')
        ne = d.groupby(['pt', 'seat']).episode.nunique()
        s = d.groupby(['pt', 'seat', 'per', 'cb', 'ph']).punishment.sum().unstack(['per', 'cb', 'ph'], fill_value=0)
        rows.append(s.div(ne, axis=0))
s = pd.concat(rows).fillna(0).groupby('pt').mean()
s.columns = ['_'.join(c) for c in s.columns]
x = pd.read_parquet('/tmp/v3_pr_pf-20261112-2-v2.parquet')
x['lead'] = x.pool_me - x.pool_ah
s = s.join(x.groupby('pt').lead.mean())
s.to_csv('scripts/policy_finder/pf-20261112-2/v3_target_value_v2.csv')
s['early'] = s.filter(like='early').sum(1)
cols = ['early'] + [c for c in s.columns if c.startswith('mid') and s[c].mean() > 0.3]
print(s[cols].mean().round(2))
X = np.column_stack([np.ones(len(s))] + [s[c] / 24 for c in cols])
b, *_ = la.lstsq(X, s.lead.values, rcond=None); r = s.lead.values - X @ b
se = np.sqrt(np.diag(la.inv(X.T @ X)) * r.var() * len(r) / (len(r) - X.shape[1]))
for n, bb, ss in zip(['const'] + cols, b, se): print(f'{n:14s} {bb:7.2f} +- {ss:5.2f}')
print(s[cols].corr().round(2))
