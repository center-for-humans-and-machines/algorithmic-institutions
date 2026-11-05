"""v4: per-round c, p, payoff of the best real final points vs the zero manager."""
import pandas as pd
P = 'plots/simulation/policy_finder/'
src = [('zero', 'plots/simulation/29_EXP1_reference/per_round.parquet', 'zero'),
       ('v2gp', P + 'pf-20261105-11-v2_sweep/final/per_round.parquet', 'v2_gp'),
       ('v3s155', P + 'pf-20261105-11-v3_sweep/final/per_round.parquet', 'v3_s155'),
       ('v3gp', P + 'pf-20261105-11-v3_sweep/final/per_round.parquet', 'v3_gp')]
out = {}
for name, path, key in src:
    d = pd.read_parquet(path); d = d[d.run.str.endswith(key)]
    v = d[d.contribution_valid]
    g = v.groupby('round_number').agg(c=('contribution', 'mean'), p=('punishment', 'mean'), pay=('payoff', 'mean'))
    out[name] = g
    print(name, d.run.iloc[0], 'sum pay', g.pay.sum().round(1), 'bill (2*p summed)', (2 * g.p).sum().round(1))
print(pd.concat(out, axis=1).round(2).to_string())
