# v7: paired effect of v6's sweep games against zero twins (/tmp/v6_Z.parquet). By gap_full / gap_six bins,
# and OLS over games clustered by point.
import numpy as np, pandas as pd, json
import statsmodels.formula.api as smf
Z=pd.read_parquet('/tmp/v6_Z.parquet')
pars={p['name']:p['params'] for p in json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261112-1-v6_sweep/sweep.json'))['points']}
fr=[]
for part in '12':
    d=np.load(f'/tmp/v6_v6_{part}.npz',allow_pickle=True)
    fr.append(pd.DataFrame(dict(part=part,row=np.arange(len(d['key'])),key=d['key'],seat=d['seat'],point=d['point'],pun=d['pun'],
                                lead=d['lead'],po=d['po'],pt=d['pt'],mem=d['mem'])))
df=pd.concat(fr); print('games',len(df))
df=df.merge(Z,on=['key','seat'],suffixes=('','_z'))
for a in ['lead','po','pt','mem']: df['d'+a]=df[a]-df[a+'_z']
print('games paired',len(df),'touched %.3f'%df.pun.mean(),'untouched identical %.3f'%(df[~df.pun].dlead.abs()<1e-6).mean())
print('all: dlead %.3f se %.3f'%(df.dlead.mean(),df.dlead.sem()),' touched: %.3f se %.3f'%(df[df.pun].dlead.mean(),df[df.pun].dlead.sem()))
df.loc[~df.pun,['dlead','dpo','dpt','dmem']]=0
print('untouched zeroed: dlead %.3f se %.3f'%(df.dlead.mean(),df.dlead.sem()))
df['gf']=df.point.map(lambda p:pars[p]['gap_full']); df['gs']=df.point.map(lambda p:pars[p]['gap_six'])
df.to_parquet('/tmp/v7_v6_paired.parquet')
df['gfb']=pd.cut(df.gf,[-.01,1,2,3,4,5,6,7]); df['gsb']=pd.cut(df.gs,[6,8,10,12,14,16,18,20])
ag=dict(n=('dlead','size'),touched=('pun','mean'),dlead=('dlead','mean'),se=('dlead','sem'),dpo=('dpo','mean'),dpt=('dpt','mean'),dmem=('dmem','mean'))
print(df.groupby('gfb',observed=True).agg(**ag).round(3))
print(df.groupby('gsb',observed=True).agg(**ag).round(3))
cl={'groups':pd.factorize(df.point)[0]}
print(smf.ols('dlead~gf+gs',df).fit(cov_type='cluster',cov_kwds=cl).summary().tables[1])
print(smf.ols('dlead~gf+I(gf**2)+gs+I(gs**2)',df).fit(cov_type='cluster',cov_kwds=cl).summary().tables[1])
