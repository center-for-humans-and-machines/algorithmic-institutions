"""Persistence: rows in the last punished round of the window (t = window-1, no punishment after),
c(t+k)-c(t) by own c bin and p(t) bin; also the others' change (spillover)."""
import numpy as np, pandas as pd
a=np.load('/tmp/pf7_arr.npz'); C,P,V,par=a['C'],a['P'],a['V'],a['par']
win=par[:,2].astype(int)
rows=[]
for w in range(2,15):
    g=np.where(win==w)[0]
    t=w-1
    for k in (1,2,4,8):
        if t+k>23: continue
        c0=C[g,:,t]; ck=C[g,:,t+k]; p0=P[g,:,t]
        ok=(V[g,:,t]*V[g,:,t+k])==1
        oc0=(C[g,:,t].sum(1,keepdims=True)-c0)/3; ock=(C[g,:,t+k].sum(1,keepdims=True)-ck)/3
        op=(P[g,:,t].sum(1,keepdims=True)-p0)/3
        rows.append(pd.DataFrame(dict(w=w,k=k,c=c0[ok],dc=(ck-c0)[ok],p=p0[ok],doc=(ock-oc0)[ok],op=np.broadcast_to(op,c0.shape)[ok])))
d=pd.concat(rows)
d['cb']=pd.cut(d.c,[-1,0,4,9,13,16,20])
d['pb']=pd.cut(d.p,[-1,0,3,7,12,20,30])
pd.set_option('display.width',250)
print(d.pivot_table(index=['cb','pb'],columns='k',values='dc',aggfunc='mean',observed=False).round(1))
print(d.pivot_table(index=['cb','pb'],columns='k',values='dc',aggfunc='size',observed=False))
# spillover: change of player's own c by others' punishment, for unpunished players
u=d[d.p==0].copy(); u['opb']=pd.cut(u.op,[-1,0,2,5,10,30])
print(u.pivot_table(index=['cb','opb'],columns='k',values='dc',aggfunc='mean',observed=False).round(1))
