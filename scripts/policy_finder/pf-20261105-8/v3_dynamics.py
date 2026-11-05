"""v3: per-round dynamics of v2's final (best) rule vs v1's final and zero."""
import pandas as pd, numpy as np
f2 = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8-v2_sweep/final/per_round.parquet')
f1 = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8_sweep/final/per_round.parquet')
z = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
z = z[z.run.str.contains('zero')]
def summ(d):
    v = d[d.contribution_valid.astype(bool)]
    g = v.groupby('round_number')
    return pd.DataFrame({'c': g.contribution.mean(), 'p': g.punishment.mean(), 'pay': g.payoff.mean(),
                         'c<5': g.contribution.apply(lambda x: (x < 5).mean()),
                         'c20': g.contribution.apply(lambda x: (x == 20).mean()),
                         'pun': g.punishment.apply(lambda x: (x > 0).mean())})
for name, d in [('zero', z), ('v1', f1[f1.run.str.endswith('_gp')]), ('v2', f2[f2.run.str.endswith('_gp')])]:
    s = summ(d); print(name); print(s.round(2).T.to_string())
    v = d[d.contribution_valid.astype(bool)]
    print('score', v.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum().mean())
