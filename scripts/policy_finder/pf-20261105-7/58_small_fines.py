"""Small fines on near-full contributors: players at c in [lo,hi] at round t, by fine at t (0,1,2-3,4-6,7+).
Own next c, own mean over rest, peers' mean over rest minus at t; all v1-v4 sweeps."""
import numpy as np
CC=[];PP=[]
for v in ['pf-20261105-7','pf-20261105-7-v2','pf-20261105-7-v3','pf-20261105-7-v4']:
    d=np.load(f'/tmp/pf7/{v}_sweep.npz'); CC.append(d['C']);PP.append(d['P'])
C=np.concatenate(CC).astype(np.float32);P=np.concatenate(PP).astype(np.float32)
for lo,hi in [(18,19),(16,17),(13,15)]:
  for t in [2,4,6,8,10,12]:
    base=(C[:,:,t]>=lo)&(C[:,:,t]<=hi)
    out=[f'c{lo}-{hi} t{t:2d}']
    for a,b in [(0,0),(1,1),(2,3),(4,6),(7,30)]:
        mm=base&(P[:,:,t]>=a)&(P[:,:,t]<=b)
        if mm.sum()<200: out.append(f' | p{a}-{b}: -'); continue
        g,i=np.nonzero(mm)
        nxt=C[g,i,t+1].mean()-C[g,i,t].mean(); rest=C[g,i,t+1:].mean()-C[g,i,t].mean()
        pe=(C[g,:,t+1:].sum(1)-C[g,i,t+1:]).mean()/3-(C[g,:,t].sum(1)-C[g,i,t]).mean()/3
        out.append(f' | p{a}-{b}: n{mm.sum():6d} d1 {nxt:+4.1f} rest {rest:+4.1f} pe {pe:+4.1f}')
    print(''.join(out))
