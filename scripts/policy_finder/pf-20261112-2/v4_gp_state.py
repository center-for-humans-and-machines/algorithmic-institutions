"""v4: in the never-punish final (gp of v3's final, seeds identical across versions),
per episode and round: my group's size, ah's size, pools; how the lead depends on sizes."""
import pandas as pd, numpy as np
df = pd.read_parquet('plots/simulation/policy_finder/pf-20261112-2-v3_sweep/final/per_round.parquet')
df = df[df.run.str.contains('_gp')]
df['seat'] = np.where(df.run.str.contains('by pf-'), 0, 1)
df['mine'] = df.group_id == df.seat
g = df.groupby(['seat', 'episode', 'round_number', 'mine']).agg(n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum'), cv=('contribution_valid','sum')).unstack('mine', fill_value=0)
g.columns = [f'{a}_{"me" if m else "ah"}' for a, m in g.columns]
g = g.reset_index()
for s in ['me', 'ah']:
    g[f'pool_{s}'] = 1.6 * g[f'c_{s}'] - g[f'p_{s}']
g['lead'] = g.pool_me - g.pool_ah
print('overall lead', g.lead.mean().round(2), 'by seat', g.groupby('seat').lead.mean().round(2).tolist())
print(g.groupby('round_number')[['n_me', 'n_ah', 'pool_me', 'pool_ah', 'p_ah', 'lead']].mean().round(2))
g['pc_me'] = g.c_me / g.n_me.replace(0, np.nan); g['pc_ah'] = g.c_ah / g.n_ah.replace(0, np.nan)
print(g.groupby('n_me')[['pc_me', 'pc_ah', 'p_ah', 'lead']].agg(['mean', 'size']).round(2))
ep = g.groupby(['seat', 'episode']).agg(lead=('lead', 'mean'), nme_end=('n_me', 'last'))
print(ep.groupby('nme_end').lead.agg(['mean', 'size']).round(2))
g.to_parquet('/tmp/v4_gp.parquet')
