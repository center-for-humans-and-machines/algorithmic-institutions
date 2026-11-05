"""v2: per-round dynamics of v1's final best point vs zero reference."""
import pandas as pd, sys
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-11_sweep/final/per_round.parquet')
d = d[d.run.str.endswith('_gp')]
v = d[d.contribution_valid]
g = v.groupby('round_number').agg(c=('contribution', 'mean'), p=('punishment', 'mean'), pay=('payoff', 'mean'),
                                  frac_p=('punishment', lambda x: (x > 0).mean()))
ref = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(ref.run.unique())
for r in ref.run.unique():
    rv = ref[(ref.run == r) & ref.contribution_valid]
    g['c_' + r.split()[-1]] = rv.groupby('round_number').contribution.mean()
print(g.round(2).to_string())
# distribution of contributions in rounds 13 and 23
for rn in [0, 1, 2, 5, 13, 14, 18, 23]:
    x = v[v.round_number == rn].contribution
    print(rn, x.value_counts(normalize=True).sort_index().round(2).to_dict())
