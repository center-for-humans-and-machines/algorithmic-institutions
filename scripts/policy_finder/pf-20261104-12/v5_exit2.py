"""v5: episode-level at decision B (cached sweeps, rule = g 0): the rule group's pool per round over the rest
of the game on counts of leavers and joiners split by their block mean contribution (low <= 5, mid 5-12,
high > 12), controlling for the block's common good of both groups, the rule group's block contribution by
band (counts of members per band) and the punishment dealt per band. Leaving is a stochastic draw given
the switch model's inputs, which these controls approximate."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
B = int(sys.argv[1]); files = sys.argv[2:]
rows = []
for f in files:
    z = np.load(f); c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); g = z['g']
    s = slice(4*B, 4*B+4); nxt = 4*B+4
    m0 = g == 0
    stay0 = (g[..., s] == 0).all(-1); stay1 = (g[..., s] == 1).all(-1)
    cb = c[..., s].mean(-1); pb = p[..., s].sum(-1)
    move = g[..., nxt] != g[..., 4*B+3]
    bands = {'lo': cb <= 5, 'mi': (cb > 5) & (cb <= 12), 'hi': cb > 12}
    r = {'pt': np.repeat(np.arange(c.shape[0]), c.shape[1]), 'f': f}
    for k, b in bands.items():
        r[f'L_{k}'] = (stay0 & b & move).sum(-1).ravel(); r[f'J_{k}'] = (stay1 & b & move).sum(-1).ravel()
        r[f'N0_{k}'] = (stay0 & b).sum(-1).ravel(); r[f'N1_{k}'] = (stay1 & b).sum(-1).ravel()
        r[f'P0_{k}'] = (pb*(stay0 & b)).sum(-1).ravel(); r[f'P1_{k}'] = (pb*(stay1 & b)).sum(-1).ravel()
    for gg in (0, 1):
        m = g == gg
        r[f'cg{gg}'] = (((1.6*c - p)*m)[..., s].sum(-2) / np.maximum(m[..., s].sum(-2), 1)).mean(-1).ravel()
    r['pool_rest'] = ((1.6*c - p)*m0)[..., nxt:].sum(-2).mean(-1).ravel()
    r['n_rest'] = m0[..., nxt:].sum(-2).mean(-1).ravel()
    r['c_rest'] = ((c*m0)[..., nxt:].sum((-1, -2)) / np.maximum(m0[..., nxt:].sum((-1, -2)), 1)).ravel()
    rows.append(pd.DataFrame(r))
d = pd.concat(rows)
d = d[(d.N0_lo + d.N0_mi + d.N0_hi) + (d.N1_lo + d.N1_mi + d.N1_hi) == 8]   # nobody moved within block (true by design)
cols = [f'{a}_{k}' for a in ['L', 'J', 'N0', 'N1', 'P0', 'P1'] for k in ['lo', 'mi', 'hi']] + ['cg0', 'cg1']
d['key'] = d.f + d.pt.astype(str)
for col in cols + ['pool_rest', 'n_rest', 'c_rest']: d[col] = d[col] - d.groupby('key')[col].transform('mean')   # within design point
X = ' + '.join(cols)
for y in ['pool_rest', 'n_rest', 'c_rest']:
    m = smf.ols(f'{y} ~ {X}', d).fit()
    print(y, {k: (round(m.params[k], 2), round(m.bse[k], 2)) for k in ['L_lo', 'L_mi', 'L_hi', 'J_lo', 'J_mi', 'J_hi']})
print('mean counts', d[[f'{a}_{k}' for a in ['L', 'J'] for k in ['lo', 'mi', 'hi']]].mean().round(3).to_dict())
