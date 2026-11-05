"""Real sweep data: value of a fine at round t on a player with contribution c: own extra contribution
summed over the remaining rounds (vs unfined players with the same c, round and prior big-fine count),
times 0.15 (payoff of a contribution point to the group mean), minus the cost 0.5 * fine.
Peers' response not counted (others' c at t+1..23 also reported)."""
import numpy as np
C=[];P=[]
for v in ['pf-20261105-7','pf-20261105-7-v2']:
    a=np.load(f'/tmp/pf7/{v}_sweep.npz'); C.append(a['C']);P.append(a['P'])
C=np.concatenate(C).astype(np.float32);P=np.concatenate(P).astype(np.float32)
nb=np.concatenate([np.zeros_like(P[...,:1]),np.cumsum(P>=10,-1)[...,:-1]],-1)
oth=(C.sum(1,keepdims=True)-C)/3
for t in [0,2,5,8,11,14,17]:
    rest=C[...,t+1:].sum(-1)-C[...,t]*(23-t)   # own gain vs staying put
    orest=oth[...,t+1:].sum(-1)-oth[...,t]*(23-t)
    print(f'round {t}: rows c-bin: fine bin -> own extra sum / value per player-fine')
    for lo,hi in [(0,3),(4,8),(9,12),(13,16),(17,19)]:
        cc=C[...,t]; m0=(cc>=lo)&(cc<=hi)&(nb[...,t]==0 if t>0 else True)
        base=rest[m0&(P[...,t]==0)]; ob=orest[m0&(P[...,t]==0)]
        if len(base)<100: continue
        row=[]
        for plo,phi in [(1,5),(6,12),(13,20),(21,30)]:
            m=m0&(P[...,t]>=plo)&(P[...,t]<=phi)
            if m.sum()<100: row.append('     n/a          '); continue
            g=rest[m].mean()-base.mean(); og=orest[m].mean()-ob.mean(); f=P[...,t][m].mean()
            row.append(f'{g:+6.1f} o{og:+5.1f} v{0.15*g-0.5*f:+5.1f}')
        print(f'  c{lo:2d}-{hi:2d} n0={len(base):6d} base {base.mean():+6.1f} |',' | '.join(row))
