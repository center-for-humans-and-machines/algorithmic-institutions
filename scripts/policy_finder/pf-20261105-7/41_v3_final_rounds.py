"""Per-round c, p, share at c<=4, payoff of v3's final points (and v2's s132)."""
import numpy as np
for v in ['pf-20261105-7-v3','pf-20261105-7-v2']:
    a=np.load(f'/tmp/pf7/{v}_final.npz'); C=a['C'].astype(float);P=a['P'].astype(float);V=a['V'].astype(float);R=a['R']
    for r in np.unique(R):
        if v.endswith('v2') and not r.endswith('s132'): continue
        m=R==r; c=C[m];p=P[m];vv=V[m]
        n=vv.sum(1)
        pay=(20-c-p+(1.6*(c*vv).sum(1,keepdims=True)-(p*vv).sum(1,keepdims=True))/np.maximum(n,1)[:,None])
        tot=((pay*vv).sum(1)/np.maximum(n,1)).sum(1)
        cm=(c*vv).sum((0,1))/vv.sum((0,1)); pm=(p*vv).sum((0,1))/vv.sum((0,1))
        low=((c<=4)*vv).sum((0,1))/vv.sum((0,1))
        print(f'{r:28s} {tot.mean():6.1f} sum p/pl {pm.sum():.1f}')
        print('  c  :',' '.join(f'{x:4.1f}' for x in cm))
        print('  p  :',' '.join(f'{x:4.1f}' for x in pm))
        print('  lo%:',' '.join(f'{100*x:4.1f}' for x in low))
