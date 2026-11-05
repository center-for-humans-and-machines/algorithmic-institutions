# v4: per design point of the v3 sweep: punishment spent per block, pool split into 1.6*sum(c) and deduction, members; regress pool on them
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v3_sweep/', 'v3')
c=d['c'].astype(np.float32); p=d['p'].astype(np.float32); m=d['mine']
gross = 1.6*(c*m).sum(1)          # [games, 24]
ded = (p*m).sum(1)
mem = m.sum(1)
rows=[]
key = pd.DataFrame(dict(pt=d['pt'], opp=d['opp']))
for (pt,o), idx in key.groupby(['pt','opp']).groups.items():
    idx=np.asarray(idx)
    r=dict(pt=pt, opp=o)
    r['pool']=(gross[idx]-ded[idx]).mean()
    r['gross']=gross[idx].mean(); r['ded']=ded[idx].mean()
    r['mem']=mem[idx].mean(); r['mem_end']=mem[idx][:,20:].mean()
    for name, sl in [('d_r0',[0]),('d_r1',[1]),('d_b1',[4,5]),('d_b2p',list(range(8,24)))]:
        r[name]=ded[idx][:,sl].sum(1).mean()
    for name, sl in [('g_b0',range(0,4)),('g_b1',range(4,8)),('g_b2',range(8,12)),('g_b3_5',range(12,24))]:
        r[name]=(gross[idx][:,list(sl)]-ded[idx][:,list(sl)]).mean()
    rows.append(r)
df=pd.DataFrame(rows)
par=d['params'].copy(); par['pt']=par.name.str.extract(r'_s(\d+)').astype(int)
df=df.merge(par.drop(columns=['pool','name']),on='pt')
df.to_csv('/tmp/pf11_v3_points.csv',index=False)
a=df.groupby('pt').mean(numeric_only=True)
print(a[['pool','gross','ded','mem','d_r0','d_r1','d_b1','d_b2p']].corr()['pool'].round(2))
import numpy.linalg as la
X=np.c_[np.ones(len(a)),a[['d_r0','d_r1','d_b1','d_b2p']].values]
b=la.lstsq(X,a.pool.values,rcond=None)[0]; print('pool ~ ded r0, r1, b1, b2+ (sum over group-rounds):',b.round(3))
# good region
g=a[(a.delta>-3)&(a.delta<0.5)&(a.rate>3.5)&(a.rate<14)]
print('n good',len(g))
X=np.c_[np.ones(len(g)),g[['d_r0','d_r1','d_b1','d_b2p']].values]
print('good region coefs', la.lstsq(X,g.pool.values,rcond=None)[0].round(3))
for o in range(1,5):
    s=df[(df.opp==o)&df.pt.isin(g.index)]
    X=np.c_[np.ones(len(s)),s[['d_r0','d_r1','d_b1','d_b2p']].values]
    print('opp',o,'coefs',la.lstsq(X,s.pool.values,rcond=None)[0].round(3), 'mean pool',s.pool.mean().round(1),'ded',s.ded.mean().round(1),'mem',s.mem.mean().round(2))
print(a.groupby(pd.cut(a.decay,[0.39,0.5,0.6,0.7,0.8,0.9,1.0,1.2]),observed=True)[['pool','gross','ded','mem','g_b0','g_b1','g_b2','g_b3_5']].mean().round(1))
