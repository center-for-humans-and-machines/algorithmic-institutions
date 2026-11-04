"""v4: per design point of cached sweeps, block-level stats of the rule's group (g==0):
members n_b, contribution of members c_b, punishment per member p_b, pool per round pool_b,
plus the other group's c. Saves /tmp/v4_points.csv and prints correlations."""
import sys, numpy as np, pandas as pd
rows = []
for f in sys.argv[1:]:
    z = np.load(f); c, p, g = (z[k].astype(np.float32) for k in 'cpg')
    keys = list(z['keys'])
    for i in range(c.shape[0]):
        r = {'src': f.split('/')[-1][:-4], 'pool': z['pool'][i], **dict(zip(keys, z['params'][i]))}
        ing = (g[i] == 0); out = (g[i] == 1)
        for b in range(6):
            sl = slice(4*b, 4*b+4)
            m = ing[..., sl]; n = m.sum(1)  # [ep, 4]
            r[f'n{b}'] = n.mean()
            r[f'c{b}'] = (c[i][..., sl]*m).sum() / max(m.sum(), 1)
            r[f'p{b}'] = (p[i][..., sl]*m).sum() / max(m.sum(), 1)
            mo = out[..., sl]
            r[f'co{b}'] = (c[i][..., sl]*mo).sum() / max(mo.sum(), 1)
            r[f'pool{b}'] = ((1.6*c[i][..., sl] - p[i][..., sl])*m).sum(1).mean()
        r['p_r0'] = (p[i][..., 0]*ing[..., 0]).sum()/ing[..., 0].sum()
        r['p_r1'] = (p[i][..., 1]*ing[..., 1]).sum()/ing[..., 1].sum()
        rows.append(r)
d = pd.DataFrame(rows); d.to_csv('/tmp/v4_points.csv', index=False)
pd.set_option('display.width', 250)
for s, x in d.groupby('src'):
    print(s); print(x.sort_values('pool', ascending=False).head(8)[['pool']+[f'n{b}' for b in range(6)]+[f'c{b}' for b in range(6)]+[f'p{b}' for b in range(5)]].round(2).to_string())
    print(x[['pool','n1','n5','c0','c1','c5','p0','p1','p2','p_r0','p_r1']].corr().round(2).to_string())
