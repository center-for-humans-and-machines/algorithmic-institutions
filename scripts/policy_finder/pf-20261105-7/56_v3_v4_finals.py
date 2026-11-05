"""Per-round comparison of the final points of v3 and v4 (c, p, payoff per round; by-point totals)."""
import numpy as np
for v in ['pf-20261105-7-v3','pf-20261105-7-v4']:
    d=np.load(f'/tmp/pf7/{v}_final.npz'); C,P,V,R=d['C'].astype(float),d['P'].astype(float),d['V'].astype(float),d['R']
    for r in np.unique(R):
        m=R==r; c=C[m];p=P[m];val=V[m]
        n=val.sum(1); pay=(20+(0.6*(c*val).sum(1)-2*(p*val).sum(1))/np.maximum(n,1))*(n>0)
        tot=pay.sum(1)
        cm=(c*val).sum((0,1))/val.sum((0,1)); pm=(p*val).sum((0,1))/val.sum((0,1))
        print(v[-3:],r[-5:],f'{tot.mean():.1f}+-{tot.std()/len(tot)**.5:.1f}','sumP',round(pm.sum(),1),'c:',' '.join(f'{x:.1f}' for x in cm[::2]),'| p:',' '.join(f'{x:.1f}' for x in pm[:12]))
