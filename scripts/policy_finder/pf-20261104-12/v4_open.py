"""v4: in the v3 sweep, does a harder opening (block-0 slope) pay at a given block-1 slope
(slope*decay)? Quadratic OLS of pool on log s0 = log slope, log s1 = log(slope*decay),
thr, peer, follow; then mean pool in cells of s0 x s1."""
import numpy as np, pandas as pd
d = pd.read_csv('/tmp/v4_points.csv'); d = d[d.src == 'v3sweep'].copy()
d['ls0'] = np.log(d.slope); d['ls1'] = np.log(d.slope*d.decay)
cols = ['ls0', 'ls1', 'thr', 'peer', 'follow']
X = [np.ones(len(d))] + [d[k] for k in cols] + [d[k]**2 for k in cols] + [d.ls0*d.ls1]
X = np.column_stack(X); b, *_ = np.linalg.lstsq(X, d.pool, rcond=None)
names = ['1'] + cols + [k+'^2' for k in cols] + ['ls0*ls1']
print(dict(zip(names, b.round(2))))
# fitted pool along ls0 at ls1 = log 5.5, other at v3 peak
for s0 in [4, 6, 8, 12, 18, 30, 50]:
    for s1 in [2, 3.5, 5.5, 8]:
        x = dict(ls0=np.log(s0), ls1=np.log(s1), thr=9.8, peer=0.47, follow=0.78)
        v = [1] + [x[k] for k in cols] + [x[k]**2 for k in cols] + [x['ls0']*x['ls1']]
        print(f's0 {s0:>3} s1 {s1:>4}: {np.dot(v, b):.1f}', end='   ')
    print()
d['b0'] = pd.cut(d.slope, [1.5, 3, 6, 12]); d['b1'] = pd.cut(d.slope*d.decay, [0, 2, 4, 7, 15])
print(d.pivot_table('pool', 'b0', 'b1', aggfunc=['mean', 'count'], observed=True).round(1))
