import sys; sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from load import load
import pandas as pd, numpy as np
df=load()
g=df.groupby(['game','player_id'])
for k in [1,2,3,4]:
    df[f'c{k}']=g.contribution.shift(-k); df[f'g{k}']=g.group_id.shift(-k)
df['cprev']=g.contribution.shift(1)
d=df[(df.player_no_input==0)&(df.contribution<=8)&(df.round_number<=18)].copy()
d['pb']=pd.cut(d.punishment,[-1,0,3,8,31])
d['cb']=pd.cut(d.contribution,[-1,2,5,8])
for k in [1,2,3,4]:
    d[f'd{k}']=np.where(d[f'g{k}']==d.group_id, d[f'c{k}']-d.contribution, np.nan)
print(d.groupby(['cb','pb'],observed=True)[['d1','d2','d3','d4']].mean().round(2))
# switching by position of punishment in block: punished in last round of block vs first rounds
df['pos']=df.round_number%4
df['block']=df.round_number//4
df['next_group']=g.group_id.shift(-1)
piv=df[df.round_number<20].pivot_table(index=['game','player_id','block'],columns='pos',values='punishment').reset_index()
sw=df[(df.pos==3)&(df.round_number<23)][['game','player_id','block','group_id','next_group']]
x=piv.merge(sw,on=['game','player_id','block']); x['switch']=(x.next_group!=x.group_id).astype(float)
import statsmodels.formula.api as smf
x.columns=[str(c) for c in x.columns]
x=x.rename(columns={'0':'p0','1':'p1','2':'p2','3':'p3'})
print(smf.logit('switch ~ p0+p1+p2+p3',data=x.dropna()).fit(disp=0).params.round(3))
# group payoff gap: pool per member own minus other
