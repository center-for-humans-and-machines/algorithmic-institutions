"""v7: group-level value of punishing at fresh round r (8, 12, 16) in v6's sweep. Episodes are grouped by
their state at r (members n, number of deep laggards c<=5 with others' mean>=10). Treated: points with
last_round >= r+2 (punish the whole phase from r); control: last_round in [r-4, r-1]. Outcome: my group's
pool summed over rounds r..23 (cost included), members at 23. OLS with state controls (sum c at r, n, n_deep,
pool sum over rounds r-4..r-1) and treated dummy, point-clustered SE approximated by point-mean regression."""
import numpy as np, json
d = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v6_sweep/sweep.json'))
LR = np.zeros(256, int); PR = {}
for p in d['points']:
    i = int(p['name'].split('_s')[-1]); LR[i] = p['params']['last_round']; PR[i] = p['params']
z = np.load('/tmp/pf/v6.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
mine = (G == 0)
pool = 1.6 * (C * mine).sum(2) - (P * mine).sum(2)
for r in [8, 12, 16]:
    m = mine[..., r] & (V[..., r] == 1); c = C[..., r]; n = m.sum(-1, keepdims=True)
    oc = ((c * m).sum(-1, keepdims=True) - c) / np.maximum(n - 1, 1)
    ndeep = (m & (c <= 5) & (oc >= 10)).sum(-1)
    tr = (LR >= r + 2); ct = (LR < r) & (LR >= r - 4)
    for k in [0, 1, 2]:
        sel = (ndeep == k) if k < 2 else (ndeep >= 2)
        res = []
        for lab, ptsel in [('T', tr), ('C', ct)]:
            s = sel & ptsel[:, None]
            res.append((lab, s.sum(), pool[..., r:].sum(-1)[s].mean(), pool[..., r - 4:r].sum(-1)[s].mean(), (P * mine)[..., r:].sum((-1, -2))[s].mean(), mine[..., 23].sum(-1)[s].mean()))
        # regression
        s = sel & (tr | ct)[:, None]
        y = pool[..., r:].sum(-1)[s]
        X = np.column_stack([np.ones(s.sum()), np.broadcast_to(tr[:, None], s.shape)[s], (C[..., r] * mine[..., r]).sum(-1)[s], mine[..., r].sum(-1)[s], pool[..., r - 4:r].sum(-1)[s]])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        print(f'r={r} n_deep={k}: ' + ' | '.join(f'{l} N={N} pool r..23 {a:.0f} (before {bf:.0f}, spend {sp:.1f}, mem23 {mm:.2f})' for l, N, a, bf, sp, mm in res) + f' | adj treat {b[1]:+.1f}')
