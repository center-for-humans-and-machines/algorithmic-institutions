"""Real sweep data: value of a big late fine on a low contributor (c<=4), rounds 8-18, matched on
round, own c, number of earlier big fines and others' mean c (bins). Also how often c<=4 happens
after round 10 under the best finals."""
import numpy as np
C=[];P=[]
for v in ['pf-20261105-7','pf-20261105-7-v2']:
    a=np.load(f'/tmp/pf7/{v}_sweep.npz'); C.append(a['C']);P.append(a['P'])
C=np.concatenate(C).astype(np.float32);P=np.concatenate(P).astype(np.float32)
nb=np.concatenate([np.zeros_like(P[...,:1]),np.cumsum(P>=10,-1)[...,:-1]],-1)
oth=(C.sum(1,keepdims=True)-C)/3
for flo in [(0,4),(5,8)]:
  print('own c in',flo)
  for t in [8,10,12,14,16,18]:
    rest=C[...,t+1:].sum(-1)-C[...,t]*(23-t); orest=(oth[...,t+1:].sum(-1)-oth[...,t]*(23-t))*3
    cc=C[...,t]; low=(cc>=flo[0])&(cc<=flo[1])
    res={}
    for fl,(plo,phi) in [('big',(15,30)),('mid',(5,14)),('small',(1,4))]:
        G=[];O=[];W=[];F=[]
        for k in [0,1,2,3]:
            nk=(nb[...,t]==k) if k<3 else (nb[...,t]>=3)
            for olo,ohi in [(0,8),(8,12),(12,16),(16,21)]:
                ok=(oth[...,t]>=olo)&(oth[...,t]<ohi)
                m0=low&nk&ok&(P[...,t]==0); m1=low&nk&ok&(P[...,t]>=plo)&(P[...,t]<=phi)
                if m0.sum()<30 or m1.sum()<30: continue
                G.append(rest[m1].mean()-rest[m0].mean()); O.append(orest[m1].mean()-orest[m0].mean()); W.append(m1.sum()); F.append(P[...,t][m1].mean())
        if W:
            W=np.array(W); g=np.average(G,weights=W); o=np.average(O,weights=W); f=np.average(F,weights=W)
            res[fl]=f'fine {f:4.1f} own {g:+6.1f} peers(sum) {o:+6.1f} v_own {0.15*g-0.5*f:+5.1f} v_all {0.15*(g+o)-0.5*f:+5.1f} n{W.sum():6d}'
    print(f'  t{t}:'); [print('     ',k,s) for k,s in res.items()]
for v,name in [('pf-20261105-7-v2','pf-20261105-7-v2_s132'),('pf-20261105-7','pf-20261105-7_gp')]:
    a=np.load(f'/tmp/pf7/{v}_final.npz'); m=a['R']==name; c=a['C'][m]; vv=a['V'][m].astype(bool)
    print(name,'share c<=4 by round (valid):',' '.join(f'{((c[...,t]<=4)&vv[...,t]).sum()/vv[...,t].sum():.3f}' for t in range(0,24,2)))
    print(name,'share c<=8 by round (valid):',' '.join(f'{((c[...,t]<=8)&vv[...,t]).sum()/vv[...,t].sum():.3f}' for t in range(0,24,2)))
