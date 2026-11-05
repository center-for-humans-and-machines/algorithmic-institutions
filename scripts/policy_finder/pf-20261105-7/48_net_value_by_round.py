"""v3 real sweep, after the opening: for players at c<=2 and c 3-6, fined (by bin) vs unfined, by round,
own mean c over the rest minus c now, and value per event in group-mean payoff:
0.15 * gain * rounds_left - 0.5 * fine (own effect only)."""
import numpy as np, json, yaml
a=np.load('/tmp/pf7/pf-20261105-7-v3_sweep.npz'); C=a['C'].astype(float);P=a['P'].astype(float);V=a['V'].astype(bool);R=a['R']
pts={p['name'].split('_')[-1]:p['params'] for p in json.load(open('plots/simulation/policy_finder/pf-20261105-7-v3_sweep/sweep.json'))['points']}
W=np.array([pts[r.split('_')[-1]]['window'] for r in R])
for clo,chi in [(0,3),(3,7)]:
    print(f'== c {clo}-{chi-1}')
    for t in range(6,22,2):
        c=C[:,:,t];p=P[:,:,t]; ok=V[:,:,t:].all(-1)&(W[:,None]<=t)
        rest=C[:,:,t+1:].mean(-1)-c; left=23-t
        line=f' t{t:2d}'
        base=None
        for plo,phi in [(0,1),(1,10),(10,20),(20,31)]:
            m=ok&(c>=clo)&(c<chi)&(p>=plo)&(p<phi)
            if m.sum()<200: line+=f'  p{plo}-{phi-1}: {"-":>16s}'; continue
            g=rest[m].mean()
            if base is None and plo==0: base=g
            val=0.15*(g-base)*left-0.5*p[m].mean() if base is not None else np.nan
            line+=f'  p{plo}-{phi-1}: {g:+5.1f} v{val:+5.1f} n{m.sum():5d}'
        print(line)
