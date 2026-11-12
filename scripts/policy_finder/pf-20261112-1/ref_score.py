import pandas as pd
from aimanager.simulation.pool_scores import add_pool, against_anchor
d=add_pool(pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet'))
print((d.agent_group==d.group_id).mean())
for m,f in against_anchor(d,'ah').items():
    n=len(f); diff=f.pool-f.other
    print(m,n,'own %.1f other %.1f lead %.2f se %.2f mem %.2f om %.2f'%(f.pool.mean(),f.other.mean(),diff.mean(),diff.std()/n**.5,f.members.mean(),f.other_members.mean()))
