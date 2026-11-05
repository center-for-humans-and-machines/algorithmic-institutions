# v4: is block-0 severity or block-1 severity what pays? quadratic fit of v3 point pools on log severity of block 0 (rate), block 1 (rate*decay), block 2 (rate*decay^2), delta, mid
import numpy as np, pandas as pd
df=pd.read_csv('/tmp/pf11_v3_points.csv')
def fit(a, label):
    s0=np.log(a.rate); s1=np.log(a.rate*a.decay); s2=np.log(a.rate*a.decay**2)
    F=dict(s0=s0,s1=s1,s2=s2,delta=a.delta,mid=a.mid)
    cols=list(F); X=[np.ones(len(a))]; names=['1']
    for i,k in enumerate(cols):
        X.append(F[k]); names.append(k)
    for i,k in enumerate(cols):
        for j,l in enumerate(cols):
            if j>=i: X.append(F[k]*F[l]); names.append(k+'*'+l)
    X=np.array(X).T
    b,res,_,_=np.linalg.lstsq(X,a.pool.values,rcond=None)
    pred=X@b; r2=1-((a.pool-pred)**2).sum()/((a.pool-a.pool.mean())**2).sum()
    print(label,'R2',round(r2,3))
    # profile: grid search over s0, s1 (s2 implied by decay) holding delta=-1.3, mid=1.2
    best=None
    for r0 in np.exp(np.linspace(np.log(2),np.log(20),30)):
        for dec in np.linspace(0.2,1.5,27):
            f=dict(s0=np.log(r0),s1=np.log(r0*dec),s2=np.log(r0*dec**2),delta=-1.3,mid=1.2)
            x=[1]+[f[k] for k in cols]+[f[k]*f[l] for i,k in enumerate(cols) for j,l in enumerate(cols) if j>=i]
            v=np.dot(x,b)
            if best is None or v>best[0]: best=(v,r0,dec)
    print('  best on grid (decay 0.2-1.5 extrapolates below 0.4)', np.round(best,2))
    return b
a=df.groupby('pt').mean(numeric_only=True)
fit(a,'all opps')
for o in range(1,5): fit(df[df.opp==o].set_index('pt'),f'opp {o}')
# raw: pool by binned block-0 and block-1 severity
a['s0b']=pd.cut(a.rate,[2,4,7,12,20]); a['s1b']=pd.cut(a.rate*a.decay,[0.5,2,3.5,6,10,25])
print(a[(a.delta>-3)&(a.delta<0.5)].pivot_table(index='s0b',columns='s1b',values='pool',aggfunc=['mean','count'],observed=True).round(1))
