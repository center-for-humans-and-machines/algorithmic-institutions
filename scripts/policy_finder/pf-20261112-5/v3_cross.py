"""Per sweep point: contribution per member-round in my group and in ah's group, members, punishment per
member-round in each; how the other group's per-member contribution moves with what I deal (cross-group herding?)."""
import numpy as np, pandas as pd, sys
z = np.load(sys.argv[1]); c, p, m, v = [z[k].astype(np.float32) for k in ['c', 'p', 'mine', 'v']]
pt = z['point'].astype(int)
o = 1 - m
rows = pd.DataFrame(dict(pt=pt,
    cm=(c * m * v).sum((1, 2)), nm=(m * v).sum((1, 2)), co=(c * o * v).sum((1, 2)), no=(o * v).sum((1, 2)),
    pm=(p * m).sum((1, 2)), po=(p * o).sum((1, 2)),
    cm4=(c * m * v)[:, :, :4].sum((1, 2)), nm4=(m * v)[:, :, :4].sum((1, 2)), co4=(c * o * v)[:, :, :4].sum((1, 2)), no4=(o * v)[:, :, :4].sum((1, 2)),
    pm4=(p * m)[:, :, :4].sum((1, 2))))
g = rows.groupby('pt').sum()
G = pd.DataFrame(dict(c_mine=g.cm / g.nm, c_oth=g.co / g.no, p_mine=g.pm / g.nm, p_oth=g.po / g.no,
                      c_mine03=g.cm4 / g.nm4, c_oth03=g.co4 / g.no4, p_mine03=g.pm4 / g.nm4))
G['pb'] = pd.cut(G.p_mine, [-0.01, 0.25, 0.5, 1, 2, 4, 30])
print(G.groupby('pb', observed=True).agg(['mean', 'size']).round(2).iloc[:, ::2])
G['pb03'] = pd.cut(G.p_mine03, [-0.01, 0.25, 0.5, 1, 2, 4, 30])
print(G.groupby('pb03', observed=True)[['c_mine03', 'c_oth03']].mean().round(2))
