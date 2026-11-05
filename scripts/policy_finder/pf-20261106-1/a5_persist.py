import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-1')
from load import load
import pandas as pd, numpy as np
df = load()
g = df.groupby(['game','player_id'])
for k in range(1,5):
    df[f'c{k}']=g.contribution.shift(-k); df[f'grp{k}']=g.group_id.shift(-k)
# only non-decision rounds where the next 3 rounds stay in same block: rounds 4k, 4k+1 (t+1,t+2 same block)
d = df[(df.player_no_input==0)&(df.round_number%4<=0)&(df.round_number<20)].copy()
d = d[(d.grp1==d.group_id)&(d.grp2==d.group_id)&(d.grp3==d.group_id)]
d['cbin']=pd.cut(d.contribution,[-1,3,8,12,20])
d['pbin']=pd.cut(d.punishment,[-1,0,3,8,30])
for k in [1,2,3]:
    d[f'd{k}']=d[f'c{k}']-d.contribution
print(d.pivot_table(index=['cbin','pbin'],values=['d1','d2','d3'],aggfunc='mean',observed=False).round(2))
print(d.pivot_table(index=['cbin','pbin'],values='d1',aggfunc='size',observed=False))
# Effect of punishing others on peer contributions: group-level, rounds 0-2 (fixed groups)
gr = df[df.round_number<3].groupby(['game','group_id','round_number']).agg(c=('contribution','mean'), p=('punishment','sum'), c1=('c1','mean')).reset_index()
import statsmodels.formula.api as smf
print(smf.ols('c1 ~ c + p',gr).fit().params.round(3).to_dict())
