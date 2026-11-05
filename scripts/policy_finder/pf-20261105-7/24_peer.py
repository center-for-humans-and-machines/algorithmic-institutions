"""Unpunished drift after the window (real sweep data): next-round change by own c and others' mean."""
import numpy as np, pandas as pd
a=np.load('/tmp/pf7_arr.npz'); C,P,V,par=a['C'],a['P'],a['V'],a['par']
win=par[:,2].astype(int)
rows=[]
for t in range(1,23):
    g=win<=t
    c=C[g,:,t]; cn=C[g,:,t+1]; oc=(C[g,:,t].sum(1,keepdims=True)-c)/3
    ok=(V[g,:,t]*V[g,:,t+1])==1
    rows.append(pd.DataFrame(dict(t=t,c=c[ok],dc=(cn-c)[ok],oc=np.broadcast_to(oc,c.shape)[ok])))
d=pd.concat(rows)
d['cb']=pd.cut(d.c,[-1,0,4,9,13,16,19,20]); d['ob']=pd.cut(d.oc,[-1,5,10,13,16,18,19.9,20])
pd.set_option('display.width',200)
print(d.pivot_table(index='cb',columns='ob',values='dc',aggfunc='mean',observed=False).round(2))
print(d.pivot_table(index='cb',columns='ob',values='dc',aggfunc='size',observed=False))
d['tb']=pd.cut(d.t,[0,8,14,19,22])
print(d[d.c==20].pivot_table(index='tb',columns='ob',values='dc',aggfunc='mean',observed=False).round(2))
