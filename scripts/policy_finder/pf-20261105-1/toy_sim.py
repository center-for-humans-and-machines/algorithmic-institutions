"""Crude linear stand-in for the artificial humans, fitted to the humanlike reference sim,
to sanity-check the rule's ranges (not a substitute for the real sweep)."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['valid']=d.contribution_valid.astype(int); d.loc[d.valid==0,'contribution']=0
d=d.sort_values(['run','episode','participant_code','round_number']); g=d.groupby(['run','episode','participant_code'])
s=d[d.valid==1].groupby(['run','episode','round_number']).contribution.agg(['sum','count']); d=d.join(s,on=['run','episode','round_number'])
d['oth']=(d['sum']-d.contribution*d.valid)/(d['count']-d.valid).clip(lower=1)
d['cn']=g.contribution.shift(-1); d['vn']=g.valid.shift(-1)
x=d[(d.valid==1)&(d.vn==1)]
c=x.contribution.values; pp=x.punishment.values
lo=(c<10); hi=(c>=18)
X=np.c_[np.ones(len(x)),c,x.oth,(pp>0)*lo,(pp>0)*(~lo)*(~hi),(pp>0)*hi,pp*lo,pp*(~lo),x.round_number]
b,res,*_=np.linalg.lstsq(X,x.cn.values,rcond=None); sd=np.std(x.cn.values-X@b)
print('coef',b.round(3),'sd',round(sd,2))
c0=d[(d.round_number==0)&(d.valid==1)].contribution.values
rng=np.random.default_rng(0)
def play(rule,n=3000):
    tot=0
    c=rng.choice(c0,(n,4)).astype(float)
    for t in range(24):
        p=np.clip(np.floor(rule(c,t)),0,30)
        tot+= (20+0.6*c.mean(1)-2*p.mean(1)).mean()
        oth=(c.sum(1,keepdims=True)-c)/3; lo=c<10; hi=c>=18
        mu=b[0]+b[1]*c+b[2]*oth+(p>0)*(b[3]*lo+b[4]*(~lo&~hi)+b[5]*hi)+p*(b[6]*lo+b[7]*(~lo))+b[8]*t
        c=np.clip(np.round(mu+rng.normal(0,sd,mu.shape)),0,20)
    return tot
def mk(T,base,k,last): return lambda c,t:(t<=last)*(c<T)*(base+k*np.clip(T-c,0,None))
print('zero',round(play(lambda c,t:0*c),1))
for T in [8,12,16,20]:
    for base in [0,1,2,4]:
        for k in [0,0.25,0.5]:
            print(T,base,k,[round(play(mk(T,base,k,L)),1) for L in [14,19,22]])
