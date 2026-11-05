"""As 49, for focal players at c 3-6 and 7-10 after the opening: own + peers' value of a fine (bins) vs unfined."""
import numpy as np, json
a=np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz'); C=a['C'].astype(float);P=a['P'].astype(float);V=a['V'].astype(bool);R=a['R']
pts={p['name'].split('_')[-1]:p['params'] for p in json.load(open('plots/simulation/policy_finder/pf-20261105-7-v3_sweep/sweep.json'))['points']}
W=np.array([pts[r.split('_')[-1]]['window'] for r in R])
for clo,chi in [(3,7),(7,11)]:
  print(f'== c {clo}-{chi-1}')
  for t in range(6,20,2):
    left=23-t
    c=C[:,:,t];p=P[:,:,t]; ok=V[:,:,t:].all(-1)&(W[:,None]<=t)
    rest=C[:,:,t+1:].mean(-1)-c; vr=V[:,:,t:].all(-1)
    prest=(rest*vr).sum(1,keepdims=True)-rest*vr
    m0=ok&(c>=clo)&(c<chi)&(p==0); g0=rest[m0].mean(); pg0=prest[m0].mean()
    line=f' t{t:2d} unf own {g0:+4.1f} peers {pg0:+4.1f} |'
    for plo,phi in [(1,10),(10,20),(20,31)]:
        m=ok&(c>=clo)&(c<chi)&(p>=plo)&(p<phi)
        if m.sum()<200: line+=f' p{plo}-{phi-1}: -'; continue
        vo=0.15*(rest[m].mean()-g0)*left-0.5*p[m].mean(); vp=0.15*(prest[m].mean()-pg0)*left
        line+=f' p{plo}-{phi-1}: own {vo:+5.1f} peers {vp:+5.1f} n{m.sum()}'
    print(line)
