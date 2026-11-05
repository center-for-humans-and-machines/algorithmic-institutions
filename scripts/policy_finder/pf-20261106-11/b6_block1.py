# v2: what in block 1 decides how many members my group holds after the first switch, per opponent; and what that is worth
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from b2_arrays import load
import statsmodels.api as sm
d=load(); c=d['c'].astype(float); p=d['p'].astype(float); g=d['g']
mg=np.where(d['mine_g0'],0,1)[:,None,None]; mine=(g==mg)
n=mine.sum(1)  # [games, 24]
pool=((1.6*c-p)*mine).sum(1)  # my pool per round
def ph(x, m, t): return (x[:,:,t]*m[:,:,t]).sum(1)/np.maximum(m[:,:,t].sum(1),1)
F=pd.DataFrame({'opp':d['opp'],'n4':n[:,4],'n8':n[:,8],'nlate':n[:,12:].mean(1),'pool':pool.mean(1)})
for t in range(4):
    F[f'pm{t}']=ph(p,mine,t); F[f'po{t}']=ph(p,~mine,t)
F['cgap']=np.mean([ph(c,mine,t)-ph(c,~mine,t) for t in range(4)],0)
for o in [1,2,3,4,0]:
    G=F if o==0 else F[F.opp==o]
    X=sm.add_constant(G[['pm0','pm1','pm2','pm3','po0','po1','po2','po3','cgap']]); r=sm.OLS(G.n4,X).fit()
    print('opp',o or 'all','n4 ~',r.params.round(3).to_dict())
r=sm.OLS(F.pool,sm.add_constant(F[['n4']])).fit(); print('pool ~ n4',r.params.round(2).to_dict())
r=sm.OLS(F.nlate,sm.add_constant(F[['n4']])).fit(); print('nlate ~ n4',r.params.round(2).to_dict())
print(F.groupby('opp')[['n4','n8','nlate','pool']].mean().round(2))
