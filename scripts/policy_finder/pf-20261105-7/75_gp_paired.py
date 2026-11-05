# The _gp finals share seeds across versions: paired per-round differences vs v3_gp.
import pandas as pd, numpy as np
base='plots/simulation/policy_finder/'
dfs=[]
for v in ['pf-20261105-7-v3','pf-20261105-7-v4','pf-20261105-7-v5','pf-20261105-7-v6']:
    d=pd.read_parquet(base+v+'_sweep/final/per_round.parquet')
    dfs.append(d[d.run.str.endswith('_gp')])
df=pd.concat(dfs); df['run']=df.run.str.replace('ah exp1 managed by pf-20261105-7-','')
v=df[df.contribution_valid]
pay=v.groupby(['run','episode','round_number']).payoff.mean().unstack('run')
con=v.groupby(['run','episode','round_number']).contribution.mean().unstack('run')
pun=v.groupby(['run','episode','round_number']).punishment.mean().unstack('run')
ref='v3_gp'
for r in ['v4_gp','v5_gp','v6_gp']:
    d=(pay[r]-pay[ref]).groupby('episode').sum()
    print(r,'total diff %.2f se %.2f'%(d.mean(),d.std()/len(d)**.5))
    dp=(pay[r]-pay[ref]).groupby('round_number').mean()
    dc=(con[r]-con[ref]).groupby('round_number').mean()
    dq=(pun[r]-pun[ref]).groupby('round_number').mean()
    print(' round  dpay   dc    dpun')
    for t in range(24): print(f'  {t:2d} {dp[t]:6.2f} {dc[t]:6.2f} {dq[t]:6.2f}')
print('v3_gp per round c / p:')
print(pd.DataFrame({'c':con[ref].groupby('round_number').mean(),'p':pun[ref].groupby('round_number').mean()}).round(2).T.to_string())
