"""Opening rounds 2-6, v2+v3 real sweeps: does the response to a fine depend on the fine the
player got last round? Own mean c over the rest minus c now, by own c bin, last fine bin, fine bin."""
import numpy as np
Cs=[];Ps=[];Vs=[]
for v in ['pf-20261105-7-v2','pf-20261105-7-v3']:
    a=np.load(f'/tmp/pf7/{v}_sweep.npz'); Cs.append(a['C']);Ps.append(a['P']);Vs.append(a['V'])
C=np.concatenate(Cs).astype(float);P=np.concatenate(Ps).astype(float);V=np.concatenate(Vs).astype(bool)
rows=[]
for t in range(2,7):
    c=C[:,:,t]; p=P[:,:,t]; pl=P[:,:,t-1]; cl=C[:,:,t-1]
    ok=V[:,:,t-1:].all(-1)
    rest=C[:,:,t+1:].mean(-1)-c; d1=C[:,:,t+1]-c
    rows.append(np.stack([c,p,pl,rest,d1,ok,cl],-1).reshape(-1,7))
X=np.concatenate(rows); X=X[X[:,5]>0]; c,p,pl,rest,d1,_,cl=X.T
for clo,chi in [(5,10),(10,15),(15,18)]:
    print(f'== own c {clo}-{chi-1}  (rest, d1)')
    for name,dm in [('last 0',pl==0),('last 1-4',(pl>=1)&(pl<5)),('last 5-9',(pl>=5)&(pl<10)),('last 10+',pl>=10)]:
        line=f'  {name:9s}'
        for plo,phi in [(0,1),(1,5),(5,10),(10,20),(20,31)]:
            m=(c>=clo)&(c<chi)&dm&(p>=plo)&(p<phi)
            if m.sum()<300: line+=f' p{plo}-{phi-1}: {"-":>17s}'; continue
            line+=f' p{plo}-{phi-1}: {rest[m].mean():+5.1f} {d1[m].mean():+5.1f} n{m.sum()//1000:3d}k'
        print(line)
