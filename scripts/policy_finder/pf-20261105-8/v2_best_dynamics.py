"""v2: per-round dynamics of v1's final (best) rule vs the zero reference."""
import pandas as pd, numpy as np
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8_sweep/final/per_round.parquet')
z = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
z = z[z.run.str.contains('zero')]
def summ(d):
    v = d[d.contribution_valid.astype(bool)]
    g = v.groupby('round_number')
    out = pd.DataFrame({'c': g.contribution.mean(), 'p': g.punishment.mean(),
                        'pay': g.payoff.mean(), 'c0': g.contribution.apply(lambda x: (x == 0).mean()),
                        'c20': g.contribution.apply(lambda x: (x == 20).mean()),
                        'pun_frac': g.punishment.apply(lambda x: (x > 0).mean())})
    return out
for name, d in [('zero', z), ('gp', f[f.run.str.endswith('_gp')])]:
    s = summ(d)
    print(name); print(s.round(2).T.to_string())
    # score: sum over rounds of per-episode mean valid payoff
    v = d[d.contribution_valid.astype(bool)]
    print('score', v.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum().mean())
