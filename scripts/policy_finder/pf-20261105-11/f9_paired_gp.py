"""v6: the five GP-peak finals play the same 2000 games (seed 1000). Paired differences between all of them,
and where (which rounds, which lineups) v4_gp gains over v5_gp."""
import numpy as np, pandas as pd
B = 'plots/simulation/policy_finder/'
S = {}; CM = {}; PM = {}; L = None
for ver in ['pf-20261105-11', 'pf-20261105-11-v2', 'pf-20261105-11-v3', 'pf-20261105-11-v4', 'pf-20261105-11-v5']:
    d = pd.read_parquet(f'{B}{ver}_sweep/final/per_round.parquet', columns=['run', 'episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid'])
    d = d[d.run.str.endswith(f'{ver}_gp')].copy()
    d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
    d = d.sort_values(['episode', 'pl', 'round_number'])
    n = len(d) // 96
    c = d.contribution.values.reshape(n, 4, 24).astype(float); p = d.punishment.values.reshape(n, 4, 24).astype(float)
    v = d.contribution_valid.values.reshape(n, 4, 24).astype(float); c = c * v; nv = v.sum(1)
    pay = 20 - c - p + ((1.6 * c.sum(1) - p.sum(1)) / np.maximum(nv, 1))[:, None, :]
    pr = (pay * v).sum(1) / np.maximum(nv, 1) * (nv > 0)
    S[ver] = pr.sum(1); CM[ver] = (c.sum(1) / np.maximum(nv, 1)); PM[ver] = p.sum(1) / 4
    if L is None: L = np.sort(np.where(v[:, :, 0] > 0, c[:, :, 0], 99), 1)[:, 0]
ref = S['pf-20261105-11-v4']
for k in S:
    dd = S[k] - ref; print(f'{k:20s} {S[k].mean():6.2f}  vs v4_gp {dd.mean():+5.2f} (paired se {dd.std()/np.sqrt(len(dd)):.2f})')
a, b = 'pf-20261105-11-v4', 'pf-20261105-11-v5'
print('v4 - v5 per round, mean c diff:', ' '.join(f'{x:+.2f}' for x in (CM[a] - CM[b]).mean(0)))
print('v4 - v5 per round, mean fine/player diff:', ' '.join(f'{x:+.2f}' for x in (PM[a] - PM[b]).mean(0)))
for lo, hi in [(0, 0), (1, 4), (5, 8), (9, 20)]:
    m = (L >= lo) & (L <= hi); dd = S[a][m] - S[b][m]
    print(f' lowest c0 {lo}-{hi}: n {m.sum()} v4-v5 {dd.mean():+.2f} (se {dd.std()/np.sqrt(m.sum()):.2f})')
