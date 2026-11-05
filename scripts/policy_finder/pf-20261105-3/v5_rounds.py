"""v5: per-round means (contribution, punishment, share of free riders, payoff) of
v4's final points and the zero reference."""
import sys, numpy as np, pandas as pd
f = sys.argv[1]
d = pd.read_parquet(f)
print(d.columns.tolist()[:30])
d['valid'] = d.contribution_valid.astype(bool)
for run, s in d.groupby('run'):
    s = s[s.valid]
    g = s.groupby('round_number')
    out = pd.DataFrame({'c': g.contribution.mean(), 'p': g.punishment.mean(),
                        'fr': g.contribution.apply(lambda x: (x <= 3).mean()),
                        'full': g.contribution.apply(lambda x: (x == 20).mean()),
                        'punished': g.punishment.apply(lambda x: (x > 0).mean())})
    if 'payoff' in s:
        out['pay'] = g.payoff.mean()
    print(run); print(out.round(2).T.to_string())
