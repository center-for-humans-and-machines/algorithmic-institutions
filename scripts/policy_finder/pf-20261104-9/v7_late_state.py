"""v7: per-round state of my group in v6's final (gp, s202) and zero_vs_ah: members, c per member,
share of members at c=20 / c<=5 / below 0.8x others' mean, pool, and ah's group c per member."""
import pyarrow.parquet as pq, pandas as pd, numpy as np
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run']
a = pq.read_table('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v6_sweep/final/per_round.parquet', columns=cols).to_pandas()
z = pq.read_table('/opt/algorithmic-institutions/plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet', columns=cols).to_pandas()
z = z[z.run.str.contains('zero_vs_ah')]
for name, d in [('gp', a[a.run.str.contains('_gp_')]), ('s202', a[a.run.str.contains('s202')]), ('zero', z)]:
    d = d.copy(); ne = d.episode.nunique()
    m = d[d.group_id == 0]; o = d[d.group_id == 1]
    m = m.assign(tot=m.groupby(['episode', 'round_number']).contribution.transform('sum'),
                 n=m.groupby(['episode', 'round_number']).contribution.transform('size'))
    m = m.assign(oc=(m.tot - m.contribution) / (m.n - 1).clip(lower=1))
    g = m.groupby('round_number')
    r = pd.DataFrame({'mem': g.size() / ne, 'c': g.contribution.mean(), 'p': g.punishment.mean(),
                      'at20': g.contribution.apply(lambda x: (x == 20).mean()), 'le5': g.contribution.apply(lambda x: (x <= 5).mean()),
                      'lag': g.apply(lambda x: (x.contribution < 0.8 * x.oc).mean()),
                      'pool': (1.6 * g.contribution.sum() - g.punishment.sum()) / ne,
                      'ah_mem': o.groupby('round_number').size() / ne, 'ah_c': o.groupby('round_number').contribution.mean(),
                      'ah_p': o.groupby('round_number').punishment.mean()})
    print(name, ne); print(r.round(2).to_string())
