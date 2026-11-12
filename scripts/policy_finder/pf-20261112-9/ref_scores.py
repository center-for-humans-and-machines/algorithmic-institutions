import pandas as pd, numpy as np, sys
sys.path.insert(0,'src')
from aimanager.simulation.pool_scores import add_pool, against_anchor
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
add_pool(d)
print((d.group_id!=d.agent_group).mean())
s=against_anchor(d,'ah')
for m,f in s.items():
    diff=f.pool-f.other
    print(m, 'own %.2f other %.2f lead %.2f se %.2f members %.2f other_members %.2f'%(f.pool.mean(),f.other.mean(),diff.mean(),diff.std()/np.sqrt(len(f)),f.members.mean(),f.other_members.mean()))
