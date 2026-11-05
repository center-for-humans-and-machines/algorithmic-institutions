"""v2: per-round view of v1's final points (each run) and of the reference runs."""
import pandas as pd, numpy as np
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-3_sweep/final/per_round.parquet')
z = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d_all = pd.concat([f, z])
for r in d_all.run.unique():
    d = d_all[d_all.run == r]
    v = d[d.contribution_valid]
    g = v.groupby('round_number')
    out = pd.DataFrame({'c': g.contribution.mean(), 'p': g.punishment.mean(),
                        'pay': g.payoff.mean(), 'c<=2': g.contribution.apply(lambda x: (x <= 2).mean()),
                        'c>=18': g.contribution.apply(lambda x: (x >= 18).mean()),
                        'p>0': g.punishment.apply(lambda x: (x > 0).mean())})
    print(r); print(out.round(2).iloc[[0,1,2,3,5,8,10,12,15,18,20,22,23]].to_string())
    print('sum of mean pay', round(out.pay.sum(), 1))
