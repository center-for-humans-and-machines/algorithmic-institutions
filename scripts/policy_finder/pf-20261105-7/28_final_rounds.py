"""Per-round contribution, punishment, payoff of the final points of v1 and v2."""
import numpy as np
for v in ['pf-20261105-7','pf-20261105-7-v2']:
    a=np.load(f'/tmp/pf7/{v}_final.npz'); C=a['C'].astype(float);P=a['P'].astype(float);V=a['V'].astype(float);R=a['R']
    print(v, 'valid frac',V.mean(), 'c range',C.min(),C.max())
    for r in np.unique(R):
        m=R==r; c=C[m];p=P[m];vv=V[m]
        n=vv.sum(1)
        pay=(20-c-p+(1.6*(c*vv).sum(1,keepdims=True)-(p*vv).sum(1,keepdims=True))/np.maximum(n,1)[:,None])
        tot=((pay*vv).sum(1)/np.maximum(n,1)).sum(1)
        cm=(c*vv).sum((0,1))/vv.sum((0,1)); pm=(p*vv).sum((0,1))/vv.sum((0,1))
        print(f' {r:28s} {tot.mean():6.1f}  c:',' '.join(f'{x:4.1f}' for x in cm[::2]),' p:',' '.join(f'{x:3.1f}' for x in pm[:12]), ' sum p/pl',round(pm.sum(),1))
