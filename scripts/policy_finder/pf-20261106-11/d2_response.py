# v3: in the v2 sweep, next-round contribution change of my members who stay, by distance to the others' mean and punishment
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load()
rng = np.random.default_rng(0); idx = rng.choice(len(d['c']), 40000, replace=False)
c = d['c'][idx].astype(float); p = d['p'][idx].astype(float); m = d['mine'][idx]; v = d['v'][idx].astype(bool)
rows = []
for t in range(23):
    ok = m[:,:,t] & v[:,:,t]
    tot = (c[:,:,t]*ok).sum(1, keepdims=True); n = ok.sum(1, keepdims=True)
    om = np.where(n-ok>0, (tot - c[:,:,t]*ok)/np.maximum(n-ok,1), np.nan)
    stay = ok & m[:,:,t+1] & v[:,:,t+1]
    gi, pi = np.nonzero(stay)
    rows.append(pd.DataFrame(dict(t=t, c=c[gi,pi,t], om=om[gi,pi], p=p[gi,pi,t], dc=c[gi,pi,t+1]-c[gi,pi,t], dc3=(c[gi,pi,min(t+3,23)]-c[gi,pi,t]))))
df = pd.concat(rows).dropna()
df['gap'] = pd.cut(df.c - df.om, [-21,-8,-5,-3,-2,-1,0,2,5,21])
df['pb'] = pd.cut(df.p, [-1,0,3,7,12,20,30])
print('mean next-round change'); print(df.pivot_table(index='gap', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
print('counts'); print(df.pivot_table(index='gap', columns='pb', values='dc', aggfunc='size', observed=True))
print('pb by c bin, gap<-2:'); df['cb']=pd.cut(df.c,[-1,0,4,8,12,16,20])
print(df[df.c-df.om<-2].pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
