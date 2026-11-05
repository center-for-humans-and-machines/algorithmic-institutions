import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from load import load
df = load().sort_values(['game','player_id','round_number'])
g = df.groupby(['game','player_id'])
df['grp_next'] = g.group_id.shift(-1)
for k in range(4):
    df[f'p{k}'] = g.punishment.shift(k)   # p0 = decision round, p3 = first of block
dec = df[df.round_number.isin([3,7,11,15,19])].copy()
dec['switch'] = (dec.grp_next!=dec.group_id).astype(int)
dec['pool']=dec.common_good
m = smf.logit('switch ~ p0+p1+p2+p3+pool', data=dec.dropna(subset=['p0','p1','p2','p3'])).fit(disp=0); print(m.summary().tables[1])
# Contribution persistence: does a punishment raise contributions for several rounds?
df['pun_any']=df.punishment
for h in [1,2,3,4]:
    df[f'cf{h}']=g.contribution.shift(-h); df[f'gf{h}']=g.group_id.shift(-h)
low = df[(df.contribution<=8)&(df.player_no_input==0)]
for h in [1,2,3,4]:
    s=low[low[f'gf{h}']==low.group_id]
    r=smf.ols(f'cf{h} ~ punishment + contribution', data=s).fit()
    print(h, 'coef punishment on c(t+%d) for c<=8:'%h, round(r.params['punishment'],3), '+-', round(r.bse['punishment'],3))
