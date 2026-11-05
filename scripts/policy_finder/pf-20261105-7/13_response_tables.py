"""dc = c(t+1)-c(t) by own c bin x punishment bin x round bin, from the sweep arrays."""
import numpy as np, pandas as pd
a=np.load('/tmp/pf7_arr.npz'); C,P,V=a['C'],a['P'],a['V']
c=C[:,:,:-1].ravel(); cn=C[:,:,1:].ravel(); p=P[:,:,:-1].ravel()
ok=(V[:,:,:-1]*V[:,:,1:]).ravel()==1
t=np.broadcast_to(np.arange(23),C[:,:,:-1].shape).ravel()
# others' mean this round and others' punishment
oc=((C.sum(1,keepdims=True)-C)/3)[:,:,:-1].ravel()
op=((P.sum(1,keepdims=True)-P)/3)[:,:,:-1].ravel()
d=pd.DataFrame(dict(c=c,cn=cn,p=p,t=t,oc=oc,op=op))[ok]
d['dc']=d.cn-d.c
d['cb']=pd.cut(d.c,[-1,0,4,9,13,16,19,20])
d['pb']=pd.cut(d.p,[-1,0,3,7,12,20,30])
d['tb']=pd.cut(d.t,[-1,0,2,6,12,22])
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(d.pivot_table(index=['tb','cb'],columns='pb',values='dc',aggfunc='mean',observed=False).round(1))
print(d.pivot_table(index=['tb','cb'],columns='pb',values='dc',aggfunc='size',observed=False))
