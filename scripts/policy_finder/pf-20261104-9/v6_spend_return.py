"""v6: across all 1280 design points (v1-v5 sweeps), how does the point's pool relate to how its punishment
was spent: per round position (0 / 1-2), per phase (0-3, 4-7, 8-11, 12+), and on deep (o-c >= 7) vs shallow
laggards and on members at/above 0.9 x others' mean. Spend = points dealt per episode. OLS with quadratic terms
summarised as the marginal return at the mean of the top-100 points."""
import numpy as np, json
vers = ['pf-20261104-9', 'pf-20261104-9-v2', 'pf-20261104-9-v3', 'pf-20261104-9-v4', 'pf-20261104-9-v5']
F, Y = [], []
for v in vers:
    pts = json.load(open(f'plots/simulation/policy_finder/{v}_sweep/sweep.json'))['points']
    pool = {int(p['name'].split('_s')[-1]): p['pool'] for p in pts}
    z = np.load(f'/tmp/pf/{v}.npz')
    C = z['C'].astype(np.float32); V = z['V'].astype(np.float32); P = z['P'].astype(np.float32); G = z['G']
    mine = (G == 0) * V
    tot = (C * mine).sum(2, keepdims=True); n = mine.sum(2, keepdims=True)
    o = (tot - C * mine) / np.maximum(n - mine, 1)
    gap = (o - C) * mine
    Pm = P * mine
    t = np.arange(24)
    feats = []
    for ph in [(0, 4), (4, 8), (8, 12), (12, 24)]:
        for pos in [[0], [1, 2]]:
            rs = [r for r in range(*ph) if r % 4 in pos]
            for cls in [gap >= 7, (gap < 7) & (C < 0.9 * o), C >= 0.9 * o]:
                feats.append((Pm[..., rs] * cls[..., rs]).sum((2, 3)).mean(1))
    feats.append(Pm[..., [r for r in range(24) if r % 4 == 3]].sum((2, 3)).mean(1))
    F.append(np.stack(feats, 1)); Y.append(np.array([pool[i] for i in range(256)]))
F = np.concatenate(F); Y = np.concatenate(Y)
names = [f'ph{a}-{b} pos{pos} {cl}' for (a, b) in [(0, 3), (4, 7), (8, 11), (12, 23)] for pos in ['0', '12'] for cl in ['deep', 'shallow', 'atmean']] + ['pre-decision']
top = np.argsort(-Y)[:100]
print('top-100 mean pool', Y[top].mean().round(1))
X = np.column_stack([np.ones(len(Y)), F, F ** 2])
b, *_ = np.linalg.lstsq(X, Y, rcond=None)
k = F.shape[1]
pred = X @ b; print('R2', 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum())
for j, nm in enumerate(names):
    m = F[top, j].mean(); marg = b[1 + j] + 2 * b[1 + k + j] * m
    print(f'{nm:24s} top-100 spend {m:6.1f}  all-mean {F[:, j].mean():6.1f}  marginal pool/100pts {marg * 100:+6.2f}')
