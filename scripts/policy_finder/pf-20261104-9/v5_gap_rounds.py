"""Round by round, the per-capita pool gap a rule could see BEFORE punishing (mine: 1.6 x mean c of my
valid members; other: 1.6 x mean c of its valid members - its mean punishment last round), and the
realised gap after punishment, for v3's final run and zero_vs_ah. Also my punishment per member."""
import pandas as pd, numpy as np, sys
for fn, run in [('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/final/per_round.parquet', None),
                ('/opt/algorithmic-institutions/plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet', 'zero_vs_ah')]:
    d = pd.read_parquet(fn, columns=['run', 'episode', 'round_number', 'group_id', 'contribution', 'punishment', 'contribution_valid'])
    runs = d.run.unique()
    r = [x for x in runs if (run is None and '_gp' in x) or (run is not None and run in x)]
    d = d[d.run == (r[0] if r else runs[0])]
    print('==', d.run.iloc[0])
    d['ci'] = d.contribution * d.contribution_valid; d['pi'] = d.punishment * d.contribution_valid
    g = d.groupby(['episode', 'round_number', 'group_id']).agg(nv=('contribution_valid', 'sum'), c=('ci', 'sum'), p=('pi', 'sum')).reset_index()
    g['pre'] = 1.6 * g.c / g.nv.clip(lower=1); g['post'] = (1.6 * g.c - g.p) / g.nv.clip(lower=1); g['pm'] = g.p / g.nv.clip(lower=1)
    w = g.pivot_table(index=['episode', 'round_number'], columns='group_id', values=['pre', 'post', 'pm']).fillna(0)
    w.columns = [f'{a}{b}' for a, b in w.columns]
    w = w.reset_index()
    w['pm1_last'] = w.groupby('episode')['pm1'].shift(1).fillna(0)
    w['gap_seen'] = w['pre0'] - (w['pre1'] - w['pm1_last'])
    w['gap_post'] = w['post0'] - w['post1']
    s = w.groupby('round_number')[['gap_seen', 'gap_post']].agg(['mean', lambda x: (x > 0).mean()])
    s[('pm0', '')] = w.groupby('round_number')['pm0'].mean()
    print(s.round(2).T.to_string())
