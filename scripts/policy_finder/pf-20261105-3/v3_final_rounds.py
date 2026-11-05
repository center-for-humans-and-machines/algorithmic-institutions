"""v3: per-round view of v2's final point (and v1's), plus zero."""
import pandas as pd, numpy as np
fs = {'v1': 'plots/simulation/policy_finder/pf-20261105-3_sweep/final/per_round.parquet',
      'v2': 'plots/simulation/policy_finder/pf-20261105-3-v2_sweep/final/per_round.parquet'}
z = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
ds = [pd.read_parquet(f) for f in fs.values()] + [z[z.run.str.contains('zero')]]
for d in ds:
    for r in d.run.unique():
        e = d[d.run == r]
        v = e[e.contribution_valid]
        g = v.groupby('round_number')
        out = pd.DataFrame({'c': g.contribution.mean(), 'p': g.punishment.mean(),
                            'pay': g.payoff.mean(), 'c<=5': g.contribution.apply(lambda x: (x <= 5).mean()),
                            'c>=18': g.contribution.apply(lambda x: (x >= 18).mean()),
                            'p>0': g.punishment.apply(lambda x: (x > 0).mean())})
        print(r); print(out.round(2).to_string())
        print('sum of mean pay', round(out.pay.sum(), 1))
