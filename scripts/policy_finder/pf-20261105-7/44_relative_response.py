"""Opening rounds 1-5, v2+v3 real sweeps: response of a player's c (t+1, t+3) and of the
others' mean c (t+3) to the fine, by own c bin and the others' mean c (relative position)."""
import numpy as np
Cs=[];Ps=[];Vs=[]
for v in ['pf-20261105-7-v2','pf-20261105-7-v3']:
    a=np.load(f'/tmp/pf7/{v}_sweep.npz'); Cs.append(a['C']);Ps.append(a['P']);Vs.append(a['V'])
C=np.concatenate(Cs).astype(float);P=np.concatenate(Ps).astype(float);V=np.concatenate(Vs).astype(bool)
G=C.shape[0]
rows=[]
for t in range(1,6):
    c=C[:,:,t]; p=P[:,:,t]; oth=(C[:,:,t].sum(1,keepdims=True)-c)/3
    ok=V[:,:,t]&V[:,:,t+1]&V[:,:,t+3]
    d1=C[:,:,t+1]-c; d3=C[:,:,t+3]-c
    od3=((C[:,:,t+3].sum(1,keepdims=True)-C[:,:,t+3])-(C[:,:,t].sum(1,keepdims=True)-c))/3
    rows.append(np.stack([np.full_like(c,t),c,p,oth,d1,d3,od3,ok],-1).reshape(-1,8))
X=np.concatenate(rows); X=X[X[:,7]>0]
t,c,p,o,d1,d3,od3=X[:,:7].T
for clo,chi in [(0,5),(5,10),(10,15),(15,18)]:
    print(f'== own c {clo}-{chi-1}')
    for olo,ohi in [(0,12),(12,16),(16,21)]:
        line=f'  others {olo:2d}-{ohi-1:2d}:'
        for plo,phi in [(0,1),(1,5),(5,10),(10,20),(20,31)]:
            m=(c>=clo)&(c<chi)&(o>=olo)&(o<ohi)&(p>=plo)&(p<phi)
            if m.sum()<300: line+=f' p{plo}-{phi-1}: {"-":>22s}'; continue
            line+=f' p{plo}-{phi-1}: {d1[m].mean():+5.1f} {d3[m].mean():+5.1f} {od3[m].mean():+5.1f} n{m.sum()//1000:3d}k'
        print(line)
