"""Distribution of contributions by round under v3's final peak; per-player trajectories by round-0 type."""
import numpy as np
a=np.load('/tmp/pf7/pf-20261105-7-v3_final.npz'); R=a['R']; m=R=='pf-20261105-7-v3_gp'
C=a['C'][m].astype(int);P=a['P'][m].astype(int);V=a['V'][m].astype(bool)
bins=[0,1,5,9,13,16,18,20,21]
for r in [0,1,3,6,10,15,20,23]:
    c=C[:,:,r][V[:,:,r]]
    h=np.histogram(c,bins=bins)[0]/len(c)
    print(r,' '.join(f'[{bins[i]},{bins[i+1]}):{100*h[i]:4.1f}' for i in range(len(h))))
# by round-0 contribution class: mean c per round, cumulative p
c0=C[:,:,0]
for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,21)]:
    s=(c0>=lo)&(c0<hi)&V[:,:,0]
    print(f'c0 in [{lo},{hi}) share {s.mean():.2f}  c:',' '.join(f'{C[s][:,r].mean():4.1f}' for r in range(0,24,2)),' sum p',P[s].sum(1).mean().round(1))
