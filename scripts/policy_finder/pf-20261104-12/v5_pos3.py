"""v5: in a cached sweep (rule = g 0), at decision after block B: logit of a rule-group member's leaving on
own punishment at block positions 0-3 and the mean punishment of his group-mates at positions 0-3, with own
block contribution, the group's gross contribution and ah's block common good as controls. Also the
joining of ah's members on the rule group's per-member punishment by position."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
B = int(sys.argv[1]); files = sys.argv[2:]
L, J = [], []
for f in files:
    z = np.load(f); c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); g = z['g']
    s = slice(4*B, 4*B+4); nxt = 4*B+4
    m0 = (g[..., s] == 0); m1 = (g[..., s] == 1)
    n0 = m0.sum(-2); n1 = m1.sum(-2)                         # [P,E,4]
    P0 = (p[..., s]*m0).sum(-2); C0 = (c[..., s]*m0).sum(-2)
    C1 = (c[..., s]*m1).sum(-2); P1 = (p[..., s]*m1).sum(-2)
    cg1 = ((1.6*C1 - P1)/np.maximum(n1, 1)).mean(-1)
    c0m = (C0/np.maximum(n0, 1)).mean(-1)
    stay0 = m0.all(-1); stay1 = m1.all(-1)
    move = g[..., nxt] != g[..., 4*B+3]
    for i in range(8):
        sel = stay0[..., i]
        own = p[..., s][..., i, :]
        mates = (P0 - own*m0[..., i, :]) / np.maximum(n0 - 1, 1)
        df = pd.DataFrame({f'own{k}': own[..., k][sel] for k in range(4)})
        for k in range(4): df[f'mate{k}'] = mates[..., k][sel]
        df['cown'] = c[..., s][..., i, :].mean(-1)[sel]; df['cgrp'] = c0m[sel]; df['cg1'] = cg1[sel]; df['n'] = n0[..., 0][sel]
        df['y'] = move[..., i][sel]; L.append(df)
        sel = stay1[..., i]
        pm = P0/np.maximum(n0, 1)
        dj = pd.DataFrame({f'pm{k}': pm[..., k][sel] for k in range(4)})
        dj['cgrp'] = c0m[sel]; dj['cg1'] = cg1[sel]; dj['ownp'] = p[..., s][..., i, :].sum(-1)[sel]; dj['n'] = n0[..., 0][sel]
        dj['y'] = move[..., i][sel]; J.append(dj)
for name, d in (('leave', pd.concat(L)), ('join', pd.concat(J))):
    if len(d) > 400000: d = d.sample(400000, random_state=0)
    X = sm.add_constant(d.drop(columns='y').astype(float))
    r = sm.Logit(d.y.astype(float), X).fit(disp=0)
    print(name, 'base rate', round(d.y.mean(), 3)); print(pd.DataFrame({'coef': r.params, 'se': r.bse}).round(4).T.to_string())
