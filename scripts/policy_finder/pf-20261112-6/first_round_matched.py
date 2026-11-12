"""v2: first-punishment-round experiment of first_round_effect.py, matched on the member's exact
contribution: next-round contribution change treated-minus-control by contribution bin x punishment size,
pooled over rounds 0-22 (staying members), with a standard error."""
import numpy as np, json
D = np.load('/tmp/v1sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6_sweep/sweep.json'))['points']}
mine = g == mg[:, None, None]
def first_allowed(q):
    if q['k'] * q['thr'] < 1: return 24
    for t in range(int(q['t_start']), 23):
        if t >= 20 or t % 4 < q['w']: return t
    return 24
fa = np.array([first_allowed(pts[n]) for n in name])
pbins = [(1, 1), (2, 3), (4, 7), (8, 30)]
cbins = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 20)]
tab = {}
for t in range(23):
    tr = fa == t; co = fa > t
    if tr.sum() == 0: continue
    m0 = mine[:, :, t] & (v[:, :, t] == 1) & (g[:, :, t + 1] == g[:, :, t]) & (v[:, :, t + 1] == 1)
    dc = c[:, :, t + 1] - c[:, :, t]
    for cv in range(21):
        cm = m0 & (c[:, :, t] == cv)
        C = cm & co[:, None]
        if C.sum() < 30: continue
        mc, vc = dc[C].mean(), dc[C].var() / C.sum()
        for i, (lo, hi) in enumerate(pbins):
            T = cm & tr[:, None] & (p[:, :, t] >= lo) & (p[:, :, t] <= hi)
            if T.sum() < 5: continue
            j = [k for k, (a, b) in enumerate(cbins) if a <= cv <= b][0]
            tab.setdefault((j, i), []).append((dc[T].mean() - mc, dc[T].var() / T.sum() + vc, T.sum()))
print('treated-minus-control next-round contribution change; rows c-bin, cols punishment bin')
print('c-bin      ' + ''.join(f'{str(b):>20s}' for b in pbins))
for j, cb in enumerate(cbins):
    s = f'{str(cb):10s} '
    for i in range(4):
        a = np.array(tab.get((j, i), [(np.nan, np.nan, 0)]))
        n = a[:, 2].sum()
        if n == 0: s += f'{"-":>20s}'; continue
        w = a[:, 2] / n; m = (w * a[:, 0]).sum(); se = np.sqrt((w ** 2 * a[:, 1]).sum())
        s += f'{m:+7.2f}({se:.2f}) n{int(n):6d}'
    print(s)
