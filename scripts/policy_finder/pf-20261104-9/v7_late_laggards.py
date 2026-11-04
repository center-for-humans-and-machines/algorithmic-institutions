"""v7: response of late deep laggards to a fresh-round punishment, v6 sweep (last_round 6..18).
At fresh round r (12, 16): my members with c <= 5 and others' mean >= 10. Treated: punished at r (points
with last_round >= r); control: same kind of member in points with last_round in [r-4, r-1]
(they stopped before r). Outcomes: c at r+1, r+3, staying at the next decision, c at 23 if in my group,
and the others' c at r+3 (herding)."""
import numpy as np, json
d = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v6_sweep/sweep.json'))
LR = np.zeros(256, int)
for p in d['points']:
    LR[int(p['name'].split('_s')[-1])] = p['params']['last_round']
z = np.load('/tmp/pf/v6.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
mine = (G == 0) & (V == 1)
for r in [8, 12, 16, 20]:
    m = mine[..., r]; c = C[..., r]; n = m.sum(-1, keepdims=True)
    oc = ((c * m).sum(-1, keepdims=True) - c) / np.maximum(n - 1, 1)
    deep = m & (c <= 5) & (oc >= 10) & (n >= 3)
    p = P[..., r]
    nxt = min(r + 3, 23); dec = (r // 4) * 4 + 4
    stay = (G[..., dec] == 0) if dec < 24 else np.ones_like(m)
    for lab, sel in [('punished(LR>=r)', (LR >= r)[:, None, None] & deep & (p > 0)),
                     ('LR>=r all', (LR >= r)[:, None, None] & deep),
                     ('control LR r-4..r-1', ((LR < r) & (LR >= r - 4))[:, None, None] & deep)]:
        if sel.sum() < 50: continue
        c1 = C[..., min(r + 1, 23)][sel]; c3 = C[..., nxt][sel]
        # others' c at r+3 among my group
        mo = mine[..., nxt]; on = (mo.sum(-1, keepdims=True) - mo)
        oth3 = ((C[..., nxt] * mo).sum(-1, keepdims=True) - C[..., nxt] * mo) / np.maximum(on, 1)
        st = stay[sel].mean()
        c23 = C[..., 23][sel & (G[..., 23] == 0)].mean()
        pool_rest = (1.6 * (C * (G == 0))[..., r:].sum((-1)) - (P * (G == 0))[..., r:].sum(-1))  # per player row: not group pool
        print(f'r={r} {lab:20s} N={sel.sum():6d} p={p[sel].mean():5.1f} c={c[sel].mean():.2f} oc={oc[sel].mean():.2f} c+1={c1.mean():.2f} c+3={c3.mean():.2f} stay={st:.3f} c23|in={c23:.2f} others@+3={oth3[sel].mean():.2f}')
