# persistence of punishment effect on low contributors, and timing (round) dependence
import sys; sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-7')
from load import load
import numpy as np, pandas as pd
df = load()
g = df.groupby(['game','round_number','group_id']).agg(gc=('contribution','mean'), n=('player_id','size')).reset_index()
df = df.merge(g, on=['game','round_number','group_id'])
df['oth']=(df.gc*df.n-df.contribution)/(df.n-1).clip(lower=1)
for k in [1,2,3,4]:
    df[f'c{k}']=df.groupby(['game','player_id']).contribution.shift(-k)
d=df[(df.player_no_input==0)&(df.round_number<20)]
d['pb']=pd.cut(d.punishment,[-1,0,3,7,30])
for lo,hi in [(-1,4),(4,9),(9,14)]:
    x=d[(d.contribution>lo)&(d.contribution<=hi)]
    print('c in',lo,hi); print(x.groupby('pb',observed=True)[['contribution','oth','c1','c2','c3','c4']].mean().round(2))
# by relative position: is it the shortfall vs group or absolute level that matters? regression for c1
import statsmodels.formula.api as smf
d=d.dropna(subset=['c1','c3'])
d['short']=(d.oth-d.contribution).clip(lower=0)
d['early']=(d.round_number<4).astype(int)
print(smf.ols('c3 ~ contribution + oth + punishment + punishment:early', d[d.contribution<=9]).fit().summary2().tables[1].round(3))
