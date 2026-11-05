# v5: across all design points of v1-v4: pool = 1.6*members*c_per_member - punishment; which part varies?
import numpy as np, pandas as pd
rows = []
for v in ['v1', 'v2', 'v3', 'v4']:
    d = np.load(f'/tmp/{v}sweep.npz'); C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']
    pt, opp, seat = d['point'], d['opp'], d['seat']
    mine = (G == seat[:, None, None]).astype(np.float32)
    n = mine.sum(1); nv = (mine * V).sum(1)
    pool = 1.6 * (C * mine * V).sum(1) - (P * mine).sum(1)
    csum = (C * mine * V).sum(1); psum = (P * mine).sum(1)
    for o in range(1, 5):
        for p_ in np.unique(pt):
            m = (pt == p_) & (opp == o)
            rows.append(dict(v=v, opp=o, pt=p_, pool=pool[m].mean(), n=n[m].mean(), nv=nv[m].mean(),
                             cpm=csum[m].sum() / nv[m].sum(), ppm=psum[m].sum() / n[m].sum(),
                             pool_b0=pool[m][:, :4].mean(), pool_late=pool[m][:, 12:].mean(),
                             n_late=n[m][:, 12:].mean(), c_late=csum[m][:, 12:].sum() / nv[m][:, 12:].sum()))
df = pd.DataFrame(rows); df.to_csv('/tmp/v5_points.csv', index=False)
for o in range(1, 5):
    s = df[df.opp == o]
    print('opp', o, 'sd pool %.2f n %.3f cpm %.3f ppm %.3f' % (s.pool.std(), s.n.std(), s.cpm.std(), s.ppm.std()))
    # linearised contributions of each component to pool variance
    mn, mc = s.n.mean(), s.cpm.mean()
    part_n = 1.6 * mc * (s.n - mn) * 0.97; part_c = 1.6 * mn * (s.cpm - mc) * 0.97; part_p = -s.ppm * s.n
    print('  sd parts: members %.2f contrib %.2f punish %.2f; corr(n,cpm) %.2f corr(n,ppm) %.2f' % (part_n.std(), part_c.std(), part_p.std(), np.corrcoef(s.n, s.cpm)[0,1], np.corrcoef(s.n, s.ppm)[0,1]))
top = df.groupby(['v','pt']).mean(numeric_only=True).sort_values('pool', ascending=False)
print(top.head(15)[['pool','n','cpm','ppm','pool_late','n_late','c_late']].round(2))
print(df.groupby('opp')[['pool','n','cpm','ppm','n_late','c_late']].mean().round(2))
