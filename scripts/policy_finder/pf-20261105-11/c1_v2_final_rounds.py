"""v3: per-round dynamics of v2's final best point (S=8) vs v1's (stop after 13)."""
import pandas as pd
out = {}
for name, path in [('v1', 'plots/simulation/policy_finder/pf-20261105-11_sweep/final/per_round.parquet'),
                   ('v2', 'plots/simulation/policy_finder/pf-20261105-11-v2_sweep/final/per_round.parquet')]:
    d = pd.read_parquet(path)
    d = d[d.run.str.endswith('_gp')]
    v = d[d.contribution_valid]
    g = v.groupby('round_number').agg(c=('contribution', 'mean'), p=('punishment', 'mean'), pay=('payoff', 'mean'),
                                      fp=('punishment', lambda x: (x > 0).mean()),
                                      c20=('contribution', lambda x: (x == 20).mean()),
                                      clo=('contribution', lambda x: (x <= 9).mean()))
    out[name] = g
    print(name, 'summed payoff', g.pay.sum().round(1), 'n eps', d.episode.nunique())
print(pd.concat(out, axis=1).round(2).to_string())
