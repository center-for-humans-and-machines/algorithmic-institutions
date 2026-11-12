"""Apply the rule's design points to the zero-managed group's contributions in the reference sims
(static: ignores responses) to see the spread of punishment per member-round the sweep covers."""
import pandas as pd, numpy as np, yaml
from aimanager.manager.rule import sobol_design
rule = yaml.safe_load(open('configs/managers/rule_based/pf-20261112-11.yml'))
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
x = d[(d.run == 'ah group_switching managed by zero_vs_ah') & (d.agent_group == 0) & (d.contribution_valid == 1)]
c, r = x.contribution.values, x.round_number.values
rows = []
for pt in sobol_design(rule, 256):
    p = np.floor(np.clip(np.minimum(pt['slope'] * np.clip(pt['threshold'] - c, 0, None), pt['cap']), 0, 30)) * (r <= pt['last_round'])
    rows.append(dict(pt, mean_p=p.mean(), frac=(p > 0).mean()))
f = pd.DataFrame(rows)
print(f.mean_p.describe().round(2)); print('share of points with mean p < 0.25:', (f.mean_p < 0.25).mean())
print('ah mean punishment per member-round (ref):', d[d.run.str.contains('ah_vs_ah')].punishment.mean().round(2))
