"""v5: what is a low giver's exit worth? Episode-level, cached sweeps (rule = g 0). Members of the rule's
group in block B (all 4 rounds) with block mean contribution <= cmax. Compare those who leave at the
block's decision with those who stay, within bins of their own block punishment and of the group's
block common good; outcomes: the rule group's pool per round over the rest of the game, its members,
and the other members' mean contribution in the next block."""
import sys, numpy as np, pandas as pd
B = int(sys.argv[1]); cmax = float(sys.argv[2]); files = sys.argv[3:]
rows = []
for f in files:
    z = np.load(f); c = z['c'].astype(np.float32); p = z['p'].astype(np.float32); g = z['g']
    P, E = c.shape[:2]
    s = slice(4*B, 4*B+4); nxt = 4*B+4
    ing = (g[..., s] == 0).all(-1)                    # in rule group whole block
    cb = c[..., s].mean(-1); pb = p[..., s].sum(-1)
    leave = g[..., nxt] == 1
    m0 = g == 0
    n_blk = m0[..., s].sum(-2).mean(-1)               # [P,E]
    cg_blk = (((1.6*c - p)*m0)[..., s].sum(-2) / np.maximum(m0[..., s].sum(-2), 1)).mean(-1)
    pool_rest = ((1.6*c - p)*m0)[..., nxt:].sum(-2).mean(-1)
    n_rest = m0[..., nxt:].sum(-2).mean(-1)
    # others' contribution next block: members in group at nxt, excluding self
    for i in range(8):
        sel = ing[..., i] & (cb[..., i] <= cmax)
        if not sel.any(): continue
        oth = np.ones(8, bool); oth[i] = False
        mo = m0[:, :, oth][..., nxt:nxt+4]; co = c[:, :, oth][..., nxt:nxt+4]
        c_oth = (co*mo).sum((-1, -2)) / np.maximum(mo.sum((-1, -2)), 1)
        pi, ei = np.nonzero(sel)
        rows.append(pd.DataFrame({'pt': pi, 'f': f, 'leave': leave[pi, ei, i], 'pb': pb[pi, ei, i], 'cb': cb[pi, ei, i],
                                  'n': n_blk[pi, ei], 'cg': cg_blk[pi, ei], 'pool_rest': pool_rest[pi, ei],
                                  'n_rest': n_rest[pi, ei], 'c_oth': c_oth[pi, ei]}))
d = pd.concat(rows)
d['pbb'] = pd.cut(d.pb, [-1, 0, 20, 40, 60, 200]); d['cgb'] = pd.cut(d.cg, [-50, 5, 10, 15, 20, 40]); d['nb'] = d.n.round()
pd.set_option('display.width', 220)
t = d.groupby(['pbb', 'leave'], observed=True)[['pool_rest', 'n_rest', 'c_oth', 'cg', 'n']].mean().round(2)
t['N'] = d.groupby(['pbb', 'leave'], observed=True).size(); print(t.to_string())
import statsmodels.formula.api as smf
for y in ['pool_rest', 'n_rest', 'c_oth']:
    r = smf.ols(f'{y} ~ leave + C(pbb) + C(cgb) + C(nb) + cb + C(pt):C(f)', d).fit()
    print(y, 'leave effect', round(r.params['leave[T.True]'], 3), 'se', round(r.bse['leave[T.True]'], 3))
