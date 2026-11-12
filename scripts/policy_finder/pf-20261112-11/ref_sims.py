"""Reference sims: zero vs ah lead, group sizes, contributions per group over rounds."""
import pandas as pd, numpy as np, sys
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
for run in ['ah group_switching managed by zero_vs_ah', 'ah group_switching managed by ah_vs_zero']:
    x = d[d.run == run]
    zg = 0 if 'by zero_vs' in run else 1
    # common_good per group per round per episode
    g = x.groupby(['episode', 'round_number', 'group_id']).agg(cg=('common_good', 'first'), n=('participant_code', 'size'),
         c=('contribution', 'mean'), p=('punishment', 'mean')).reset_index()
    full = g.pivot_table(index=['episode', 'round_number'], columns='group_id', values='cg', fill_value=0)
    ep = full.groupby('episode').mean()
    lead = ep[zg] - ep[1 - zg]
    print(run, 'zero lead', lead.mean(), '+-', lead.std() / np.sqrt(len(lead)), 'zero pool', ep[zg].mean(), 'ah pool', ep[1 - zg].mean())
    g['mgr'] = np.where(g.group_id == zg, 'zero', 'ah')
    t = g.groupby(['round_number', 'mgr']).agg(n=('n', 'mean'), c=('c', 'mean'), p=('p', 'mean'), cg=('cg', 'mean')).unstack()
    print(t.round(2).to_string())
