import pandas as pd, numpy as np
from aimanager.simulation.pool_scores import add_pool, against_anchor
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
add_pool(d)
print((d.agent_group!=d.group_id).mean())
s=against_anchor(d,'ah')
for m,f in s.items():
    lead=f.pool-f.other
    print(m, f.mean().round(2).to_dict(), 'lead %.2f se %.2f'%(lead.mean(), lead.std()/np.sqrt(len(lead))))
