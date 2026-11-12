# v6: paired effect of v5's sweep games against zero twins (/tmp/v6_Z.parquet, from v6_twins.py).
# Per design point: mean paired lead change over all its games (unpunished ones count 0), by bar and amount.
import numpy as np, pandas as pd, json
import statsmodels.formula.api as smf
Z=pd.read_parquet('/tmp/v6_Z.parquet')
pars={p['name']:p['params'] for p in json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261112-1-v5_sweep/sweep.json'))['points']}
fr=[]
for part in '12':
    d=np.load(f'/tmp/v6_v5_{part}.npz',allow_pickle=True)
    fr.append(pd.DataFrame(dict(part=part,row=np.arange(len(d['key'])),key=d['key'],seat=d['seat'],point=d['point'],pun=d['pun'],
                                lead=d['lead'],po=d['po'],pt=d['pt'],mem=d['mem'])))
df=pd.concat(fr).merge(Z,on=['key','seat'],suffixes=('','_z'))
for a in ['lead','po','pt','mem']: df['d'+a]=df[a]-df[a+'_z']
print('games paired',len(df),'touched %.3f'%df.pun.mean(),'untouched identical %.3f'%(df[~df.pun].dlead.abs()<1e-6).mean())
print('all: dlead %.3f se %.3f'%(df.dlead.mean(),df.dlead.sem()),' touched: %.3f se %.3f'%(df[df.pun].dlead.mean(),df[df.pun].dlead.sem()))
df.loc[~df.pun,['dlead','dpo','dpt','dmem']]=0
df['bar']=df.point.map(lambda p:pars[p]['bar']); df['amount']=df.point.map(lambda p:pars[p]['amount'])
df.to_parquet('/tmp/v6_v5_paired.parquet')
print(df.groupby('bar').agg(n=('dlead','size'),touched=('pun','mean'),dlead=('dlead','mean'),se=('dlead','sem'),dpo=('dpo','mean'),dpt=('dpt','mean')).round(3))
df['ab']=pd.cut(df.amount,[11,14,17,20,23,26,30])
print(df.groupby('ab',observed=True).agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem'),dpo=('dpo','mean'),dpt=('dpt','mean')).round(3))
print(df.pivot_table(index='bar',columns='ab',values='dlead',aggfunc='mean',observed=True).round(2))
print(smf.ols('dlead~bar+amount',df).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(df.point)[0]}).summary().tables[1])
print(smf.ols('dlead~bar+amount+I(amount**2)',df).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(df.point)[0]}).summary().tables[1])
