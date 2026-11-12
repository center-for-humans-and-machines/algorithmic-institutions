"""Sanity: recompute each v1 point's score from the compact arrays; then per-point drivers of the score."""
import numpy as np, json
z = np.load('/tmp/v1_compact.npz'); c, p, m, pt = z['c'].astype(float), z['p'].astype(float), z['mine'].astype(float), z['point']
P = z['params']
own = (1.6 * c * m - p * m).sum(1).mean(-1); oth = (1.6 * c * (1 - m) - p * (1 - m)).sum(1).mean(-1)
lead = own - oth
sw = json.load(open('plots/simulation/policy_finder/pf-20261112-5_sweep/sweep.json'))
ref = {int(q['name'].split('_s')[-1]): q['pool'] for q in sw['points']}
L = np.array([lead[pt == i].mean() for i in range(256)])
print('max abs diff vs sweep.json', np.abs(L - np.array([ref[i] for i in range(256)])).max())
size = m.sum(1).mean(-1); pun = (p * m).sum(1).mean(-1) / np.maximum(m.sum(1).mean(-1), 1)
cm = (c * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)
co = (c * (1 - m)).sum((1, 2)) / np.maximum((1 - m).sum((1, 2)), 1)
S = np.array([[size[pt == i].mean(), pun[pt == i].mean(), cm[pt == i].mean(), co[pt == i].mean(), L[i]] for i in range(256)])
o = np.argsort(-S[:, 1])
print('points by punishment per member-round: [k thr w end] size pun c_mine c_other lead')
for i in list(o[::16]) + [o[-1]]:
    print(np.round(P[i], 2), np.round(S[i], 2))
# bins of punishment intensity
for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, 1), (1, 2), (2, 4), (4, 30)]:
    s = (S[:, 1] >= lo) & (S[:, 1] < hi)
    print(f'pun [{lo},{hi}) n={s.sum()}', np.round(S[s].mean(0), 2))
