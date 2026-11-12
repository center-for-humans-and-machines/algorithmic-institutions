"""v2: regressions / tables on the decision-round records from v2_switch_drivers.py."""
import pandas as pd, numpy as np, numpy.linalg as la
x = pd.read_parquet('/tmp/v2_switch.parquet').dropna()
for who in ['leave_mine', 'leave_ah']:
    y = x[x.who == who]
    cols = ['c', 'cblk', 'gap', 'gap_blk', 'p0', 'p1', 'p2', 'p3', 'nme']
    X = np.column_stack([np.ones(len(y))] + [y[k].values for k in cols] + [(y.dr == k).values for k in [7, 11, 15, 19]])
    b = la.lstsq(X, y.sw.values, rcond=None)[0]
    print(who, 'n', len(y), 'mean', y.sw.mean().round(3))
    print(dict(zip(['const'] + cols + ['d7', 'd11', 'd15', 'd19'], b.round(4))))
    y = y.assign(gb=pd.cut(y.gap, [-99, -10, -5, -2, 0, 2, 5, 10, 99]), nb=y.nme)
    print(y.pivot_table(index='gb', columns='nb', values='sw', aggfunc='mean', observed=True).round(3))
    print(y.groupby('gb', observed=True).sw.agg(['mean', 'size']).round(3).T)
