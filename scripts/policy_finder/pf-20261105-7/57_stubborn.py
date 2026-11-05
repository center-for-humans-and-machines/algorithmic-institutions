"""Stubborn defectors: players at c<=2 at t and at t-1 who were fined >=15 at t-1 (v2-v4 sweeps).
Response by the fine at t: own c at t+1, own mean over the rest, peers' mean change over the rest."""
import numpy as np
for v in ['pf-20261105-7-v2','pf-20261105-7-v3','pf-20261105-7-v4']:
    d=np.load(f'/tmp/pf7/{v}_sweep.npz'); C,P=d['C'].astype(float),d['P'].astype(float)
    if v.endswith('v2'): CC,PP=C,P
    else: CC=np.concatenate([CC,C]);PP=np.concatenate([PP,P])
C,P=CC,PP
print('share of player-rounds (t 2-10) at c<=2 & prev c<=2 & prev fine>=15:')
for t in range(2,16):
    m=(C[:,:,t]<=2)&(C[:,:,t-1]<=2)&(P[:,:,t-1]>=15)
    m2=(C[:,:,t]<=2)&(C[:,:,t-1]<=2)&(P[:,:,t-1]>=15)&(P[:,:,t-2]>=15)&(C[:,:,t-2]<=2)
    out=[f't{t:2d} n={m.sum():6d} ({m.mean()*100:.2f}%) n2={m2.sum():5d}']
    for lo,hi in [(0,0),(1,9),(10,19),(20,30)]:
        mm=m&(P[:,:,t]>=lo)&(P[:,:,t]<=hi)
        if mm.sum()<30: out.append(f'  p{lo}-{hi}: -'); continue
        g,i=np.nonzero(mm)
        nxt=C[g,i,t+1].mean(); rest=C[g,i,t+1:].mean()
        peers=(C[g,:,t+1:].sum(1)-C[g,i,t+1:]).mean()/3 - (C[g,:,t].sum(1)-C[g,i,t]).mean()/3
        out.append(f'  p{lo}-{hi}: n{mm.sum():5d} next {nxt:4.1f} rest {rest:4.1f} peersΔ {peers:+.1f}')
    print(''.join(out))
