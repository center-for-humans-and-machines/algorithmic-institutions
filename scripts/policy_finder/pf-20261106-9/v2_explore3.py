# Within-cell comparison: next-round contribution vs own punishment holding own c, rounded group
# mean, round and previous own punishment fixed (punishment varies across the sweep's params).
import numpy as np, pandas as pd
d = np.load('/tmp/v1sweep.npz')
c, p, v, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['v'], d['g'], d['ours']
ing = (g == ours[:, None, None])
cnt = ing.sum(1, keepdims=True)
gm = (c * ing).sum(1, keepdims=True) / np.maximum(cnt, 1)
# others' punishment in our group this round (mean over other members)
gp = (p * ing).sum(1, keepdims=True)
rows = []
for t in range(23):
    if t % 4 == 3: continue
    m = ing[:, :, t] & ing[:, :, t + 1] & v[:, :, t] & v[:, :, t + 1]
    pp = p[:, :, t - 1] if t > 0 else np.zeros_like(p[:, :, 0])
    oth = (gp[:, 0, t][:, None] - p[:, :, t]) / np.maximum(cnt[:, 0, t][:, None] - 1, 1)
    rows.append(pd.DataFrame(dict(c=c[:, :, t][m].astype(np.int8), p=p[:, :, t][m], cn=c[:, :, t + 1][m],
        gm=np.round(np.broadcast_to(gm[:, :, t], m.shape)[m]).astype(np.int8), t=np.int8(t),
        pp=np.minimum(pp[m], 3).astype(np.int8), oth=oth[m])))
D = pd.concat(rows, ignore_index=True); del rows
D['pb'] = pd.cut(D.p, [-1, 0, 1, 2, 4, 7, 12, 20, 30], labels=['0', '1', '2', '3-4', '5-7', '8-12', '13-20', '21-30'])
D['cell'] = D.groupby(['c', 'gm', 't', 'pp']).ngroup()
D['dcn'] = D.cn - D.groupby('cell').cn.transform('mean')
for cb, (lo, hi) in {'c 0-4': (0, 4), 'c 5-9': (5, 9), 'c 10-14': (10, 14), 'c 15-20': (15, 20)}.items():
    s = D[(D.c >= lo) & (D.c <= hi)]
    # only cells with both punished and unpunished
    has = s.groupby('cell').p.agg(lambda x: (x > 0).any() and (x == 0).any())
    s = s[s.cell.map(has)]
    r = s.groupby('pb', observed=True).dcn.agg(['mean', 'size'])
    print(cb, ' '.join(f'{i}:{a:+.2f}({n})' for i, (a, n) in r.iterrows()))
# spillover: unpunished players, effect of others' mean punishment
s = D[D.p == 0].copy()
s['ob'] = pd.cut(s.oth, [-1, 0, 1, 3, 6, 10, 31])
s['dcn'] = s.cn - s.groupby('cell').cn.transform('mean')
print('unpunished, by others mean punishment:', s.groupby('ob', observed=True).dcn.agg(['mean', 'size']).round(2).to_string())
