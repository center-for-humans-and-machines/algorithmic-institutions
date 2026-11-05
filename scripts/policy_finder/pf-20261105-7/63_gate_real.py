"""Real-sweep check of the group gate (v3+v4 sweeps): players at c in [lo,hi] at round t (4-9), split by
whether c is above the group mean (incl. self) + 1, fined (p>=1, opening still on) vs unfined (window over).
Value = 0.15*(own rest gain + peers' summed rest gain)*rounds left - 0.5*fine, gains relative to the unfined."""
import numpy as np
CC=[];PP=[]
for v in ['pf-20261105-7-v3','pf-20261105-7-v4']:
    d=np.load(f'/tmp/pf7/{v}_sweep.npz'); CC.append(d['C']);PP.append(d['P'])
C=np.concatenate(CC).astype(np.float32);P=np.concatenate(PP).astype(np.float32)
for lo,hi in [(5,9),(10,13),(14,15)]:
  for t in [4,5,6,7,8,9]:
    gm=C[:,:,t].mean(1,keepdims=True)
    out=[f'c{lo:2d}-{hi} t{t}']
    for side,sm in [('above',C[:,:,t]>=gm+1),('below',C[:,:,t]<gm+1)]:
        base=(C[:,:,t]>=lo)&(C[:,:,t]<=hi)&sm
        r={}
        for k,mm in [('f',base&(P[:,:,t]>=1)),('u',base&(P[:,:,t]==0))]:
            g,i=np.nonzero(mm)
            if len(g)<200: r=None;break
            own=(C[g,i,t+1:].mean(1)-C[g,i,t]).mean()
            pe=((C[g,:,t+1:].sum(1)-C[g,i,t+1:]).mean(1)-(C[g,:,t].sum(1)-C[g,i,t])).mean()
            r[k]=(own,pe,P[g,i,t].mean(),len(g))
        if r is None: out.append(f' | {side}: -'); continue
        L=23-t
        val=0.15*((r['f'][0]-r['u'][0])+(r['f'][1]-r['u'][1]))*L-0.5*r['f'][2]
        out.append(f" | {side}: nf{r['f'][3]:5d} nu{r['u'][3]:5d} fine {r['f'][2]:4.1f} own {r['f'][0]-r['u'][0]:+4.1f} peers {r['f'][1]-r['u'][1]:+4.1f} value {val:+5.1f}")
    print(''.join(out))
