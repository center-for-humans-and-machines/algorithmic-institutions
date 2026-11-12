"""Draw the sweep design and see how much of it is (near) zero punishment;
estimate each point's punishment budget on the zero-managed groups of the reference sims."""
import pandas as pd, numpy as np
from aimanager.manager.rule import read_rule, sobol_design
import yaml; r=yaml.safe_load(open('configs/managers/rule_based/pf-20261112-12.yml'))
D=pd.DataFrame(sobol_design(r,256))
print(D.describe().round(2))
print(pd.crosstab(D.quiet,D.min_left))
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
z=d[d.run.str.contains('zero_vs_zero')&d.contribution_valid]
cs=z.contribution.values; t=z.round_number.values
budget=[]
for _,q in D.iterrows():
    gate=((t<20)&(t%4<4-q.quiet))|((t>=20)&(23-t>=q.min_left))
    p=np.floor(np.clip(q.k*np.clip(q.thr-cs,0,None),0,30))*gate
    budget.append(p.mean())
D['mean_p']=budget
print(D.mean_p.describe()); print((D.mean_p<0.05).mean())
