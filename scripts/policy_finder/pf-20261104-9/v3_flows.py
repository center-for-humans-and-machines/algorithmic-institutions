"""Membership flows at each decision in v3's final (gp point) and in zero_vs_ah: share of my group's
members leaving (split by whether they were punished in the phase), share of ah's members joining."""
import pandas as pd, numpy as np
def load(fn, key):
    d = pd.read_parquet(fn, columns=['run', 'episode', 'participant_code', 'round_number', 'group_id', 'punishment', 'contribution'])
    d = d[d.run.str.contains(key)]
    d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
    d = d.sort_values(['episode', 'pl', 'round_number'])
    n = len(d) // 24
    return d.group_id.values.reshape(n, 24), d.punishment.values.reshape(n, 24), d.contribution.values.reshape(n, 24)
for name, fn, key, mine in [('v3 gp', '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/final/per_round.parquet', 'v3_gp_vs', 0),
                            ('zero', '/opt/algorithmic-institutions/plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet', 'zero_vs_ah', 0)]:
    G, P, C = load(fn, key)
    print(name)
    for t in (3, 7, 11, 15, 19):
        inm = G[:, t] == mine; leave = inm & (G[:, t + 1] != mine); join = (~inm) & (G[:, t + 1] == mine)
        pun = P[:, t - 3:t + 1].sum(1) > 0
        print(f'  after r{t}: members {inm.reshape(-1,8).sum(1).mean():.2f} leave {leave.sum()/inm.sum():.3f} '
              f'(punished {leave[inm&pun].mean() if (inm&pun).any() else float("nan"):.3f} n/ep {(inm&pun).reshape(-1,8).sum(1).mean():.2f}, '
              f'unpunished {leave[inm&~pun].mean():.3f}) join {join.sum()/(~inm).sum():.3f}  '
              f'c leavers {C[leave, t].mean():.1f} joiners {C[join, t].mean():.1f}')
