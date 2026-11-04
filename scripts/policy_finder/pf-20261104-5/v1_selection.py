"""v2: late gain of v1's peak: memory (same players give more) or selection (who is in the group)?
Players typed by their round-0 contribution (before any punishment). Our group = group 0."""
import pandas as pd, numpy as np
F = 'plots/simulation/policy_finder/pf-20261104-5_sweep/final/per_round.parquet'
R = 'plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'
d = pd.read_parquet(F)
d = d[d.run.str.contains('_gp_')]
r = pd.read_parquet(R)
r = r[r.run.str.contains('zero_vs_ah')]
for name, x in [('v1 peak', d), ('zero', r)]:
    x = x.sort_values(['episode', 'participant_code', 'round_number']).copy()
    k = ['episode', 'participant_code']
    c0 = x[x.round_number == 0].set_index(k).contribution
    v0 = x[x.round_number == 0].set_index(k).contribution_valid
    typ = pd.cut(c0.where(v0), [-1, 4, 9, 14, 20], labels=['0-4', '5-9', '10-14', '15-20'])
    x = x.join(typ.rename('typ'), on=k)
    x['start0'] = x.join((x[x.round_number == 0].set_index(k).group_id == 0).rename('s0'), on=k).s0
    ing = x[x.group_id == 0].groupby(k).size()
    always = (ing == 24).rename('always')
    x = x.join(always, on=k); x['always'] = x.always.fillna(False)
    late = x[(x.round_number >= 12) & (x.group_id == 0) & x.contribution_valid]
    print('==', name, 'n players-rounds late', len(late))
    print('late composition by type (share of member-rounds) and mean c:')
    print(late.groupby('typ', observed=True).contribution.agg(['size', 'mean']).assign(share=lambda t: t['size'] / t['size'].sum()).round(3))
    print('late, stayers (in group 0 all 24 rounds) vs others:')
    print(late.groupby(['always', 'typ'], observed=True).contribution.agg(['size', 'mean']).round(2))
    # where do the starting members of type 0-4 spend rounds 12-23
    s = x[(x.round_number >= 12) & (x.start0 == True)]
    print('starting members: share of rounds 12-23 in group 0 by type')
    print(s.groupby('typ', observed=True).group_id.apply(lambda g: (g == 0).mean()).round(3))
