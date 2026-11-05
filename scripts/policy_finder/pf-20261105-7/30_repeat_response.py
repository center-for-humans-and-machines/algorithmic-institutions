"""Real sweep data (v1+v2): response of low contributors to a big fine by how many big fines
they already had; also the counterfactual (similar history, small/no fine)."""
import numpy as np
C=[];P=[];V=[]
for v in ['pf-20261105-7','pf-20261105-7-v2']:
    a=np.load(f'/tmp/pf7/{v}_sweep.npz'); C.append(a['C']);P.append(a['P']);V.append(a['V'])
C=np.concatenate(C).astype(float);P=np.concatenate(P).astype(float);V=np.concatenate(V).astype(bool)
C[~V]=np.nan
nbig=np.concatenate([np.zeros_like(P[...,:1]),np.cumsum(P>=10,-1)[...,:-1]],-1)  # big fines before round t
for crange in [(0,4),(5,9),(10,14)]:
  print('c in',crange)
  for t0,t1 in [(0,0),(1,3),(4,8)]:
    for nb in [0,1,2,3]:
        row=[]
        for plo,phi in [(0,0),(1,9),(10,19),(20,30)]:
            sel=np.zeros_like(C,dtype=bool)
            for t in range(t0,t1+1):
                s=(C[...,t]>=crange[0])&(C[...,t]<=crange[1])&(P[...,t]>=plo)&(P[...,t]<=phi)&((nbig[...,t]==nb) if nb<3 else (nbig[...,t]>=3))
                sel[...,t]=s
            idx=np.nonzero(sel)
            if len(idx[0])<50: row.append('   n/a          '); continue
            g,pl,t=idx
            d1=np.nanmean(C[g,pl,t+1]-C[g,pl,t]); d4=np.nanmean(C[g,pl,np.minimum(t+4,23)]-C[g,pl,t])
            row.append(f'{d1:+5.1f}/{d4:+5.1f} n{len(g):6d}')
        print(f'  t{t0}-{t1} prior_big={nb}{"+" if nb==3 else " "}', ' | '.join(row))
