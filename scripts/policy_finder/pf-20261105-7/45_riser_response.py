"""Opening rounds 2-6, v2+v3 real sweeps: does a fine still move a player who is already rising?
Own change by t+3 and own mean c over the rest of the game minus c now, by own c bin,
by direction since last round (rose >=3 / flat -2..2 / fell <=-3), by fine bin."""
import numpy as np
Cs=[];Ps=[];Vs=[]
for v in ['pf-20261105-7-v2','pf-20261105-7-v3']:
    a=np.load(f'/tmp/pf7/{v}_sweep.npz'); Cs.append(a['C']);Ps.append(a['P']);Vs.append(a['V'])
C=np.concatenate(Cs).astype(float);P=np.concatenate(Ps).astype(float);V=np.concatenate(Vs).astype(bool)
rows=[]
for t in range(2,7):
    c=C[:,:,t]; p=P[:,:,t]; dprev=c-C[:,:,t-1]
    ok=V[:,:,t-1:].all(-1)
    d3=C[:,:,t+3]-c; rest=C[:,:,t+1:].mean(-1)-c
    rows.append(np.stack([c,p,dprev,d3,rest,ok],-1).reshape(-1,6))
X=np.concatenate(rows); X=X[X[:,5]>0]; c,p,dp,d3,rest=X[:,:5].T
for clo,chi in [(5,10),(10,15),(15,18)]:
    print(f'== own c {clo}-{chi-1}')
    for name,dm in [('rose>=3',dp>=3),('flat',(dp>-3)&(dp<3)),('fell<=-3',dp<=-3)]:
        line=f'  {name:9s}'
        for plo,phi in [(0,1),(1,5),(5,10),(10,20),(20,31)]:
            m=(c>=clo)&(c<chi)&dm&(p>=plo)&(p<phi)
            if m.sum()<300: line+=f' p{plo}-{phi-1}: {"-":>17s}'; continue
            line+=f' p{plo}-{phi-1}: {d3[m].mean():+5.1f} {rest[m].mean():+5.1f} n{m.sum()//1000:3d}k'
        print(line)
