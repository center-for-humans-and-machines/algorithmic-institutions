"""v7: per-round breakdown (contribution, punishment, payoff, valid) of v6 final gp vs reference zero/humanlike."""
import pandas as pd, numpy as np
def br(d, name):
    d = d[d.contribution_valid.astype(bool)]
    g = d.groupby('round_number').agg(c=('contribution', 'mean'), p=('punishment', 'mean'), pay=('payoff', 'mean'), z=('contribution', lambda x: (x == 0).mean()))
    ep = d.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum().mean()
    print(name, 'score~', round(ep, 1))
    print(g.round(2).T.to_string())
f = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-8-v6_sweep/final/per_round.parquet')
br(f[f.run.str.endswith('_gp')], 'v6 gp')
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(r.run.unique())
for k, x in r.groupby('run'): br(x, k)
