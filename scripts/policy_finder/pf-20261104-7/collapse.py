"""Final runs of v1/v2 and reference: distribution of my group's size per switch block, and the pool
per episode by the size my group fell to after the first switch."""
import pandas as pd
from aimanager.simulation.pool_scores import add_pool, pairing_of
pd.set_option('display.width', 250)
B = 'plots/simulation/policy_finder/'
for f in [B + 'pf-20261104-7_sweep/final/per_round.parquet', B + 'pf-20261104-7-v2_sweep/final/per_round.parquet',
          'plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet']:
    df = add_pool(pd.read_parquet(f))
    for run in df.run.unique():
        g0, g1 = pairing_of(run)
        if g1 != 'ah' or g0 not in ('pf-20261104-7_gp', 'pf-20261104-7-v2_gp', 'zero'): continue
        s = df[(df.run == run)]
        n = s[s.agent_group == 0].groupby(['episode', 'round_number']).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
        pool = s[s.agent_group == 0].groupby(['episode', 'round_number']).pool.sum().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
        print(g0)
        for r in [4, 8, 12, 16, 20]:
            print('  round', r, 'size share', n[r].value_counts(normalize=True).sort_index().round(3).to_dict())
        ep = pd.DataFrame({'n4': n[4], 'pool': pool.mean(axis=1), 'late_n': n[[12, 16, 20]].mean(axis=1)})
        print(ep.groupby('n4').agg(share=('pool', 'size'), pool=('pool', 'mean'), late_n=('late_n', 'mean')).assign(share=lambda x: x.share / len(ep)).round(2))
