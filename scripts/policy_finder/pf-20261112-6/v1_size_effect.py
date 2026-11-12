"""v3: does the size-dependence of punishment's cost (blockstart_iv.py) hold across rounds in v1's sweep? Natural
experiment of the first punishable round fa (v1 points: before fa the group is run like zero). For rounds t grouped
by phase, treated = fa == t, control = fa > t. Rebuild what the rule would deal at t, Pw = sum floor(k*(thr-c)+),
and regress on T*Pw by own size at t (controls Pw by size, state). Outcomes: own size after the next decision,
lead over rounds t..t+3 (per 24 rounds), own contributions over t+1..t+3."""
import sys, numpy as np, json
sys.path.insert(0, 'scripts/policy_finder/pf-20261112-6')
import os
if not os.path.exists('/tmp/v1sweep.npz'):
    from load_sweep import load
    np.savez('/tmp/v1sweep.npz', **load('plots/simulation/policy_finder/pf-20261112-6_sweep/'))
D = np.load('/tmp/v1sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6_sweep/sweep.json'))['points']}
def first_allowed(q):
    for t in range(int(q['t_start']), 23):
        if t >= 20 or t % 4 < q['w']:
            return t
    return 24
fa = np.array([first_allowed(pts[n]) if pts[n]['k'] * pts[n]['thr'] >= 1 else 24 for n in name])
K = np.array([pts[n]['k'] for n in name]); THR = np.array([pts[n]['thr'] for n in name])
mine = g == mg[:, None, None]
own_pool = (1.6 * c * mine - p * mine).sum(1); oth_pool = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1).astype(float); own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
groups = {'n<=4': lambda s: s <= 4, 'n=5': lambda s: s == 5, 'n=6': lambda s: s == 6, 'n>=7': lambda s: s >= 7}
phases = {'r1-2': [1, 2], 'dec3': [3], 'bstart4/8': [4, 8], 'bstart12/16': [12, 16], 'mid(5,6,9,10,13,14,17,18)': [5, 6, 9, 10, 13, 14, 17, 18],
          'dec7-19': [7, 11, 15, 19], 'late20-22': [20, 21, 22]}
for ph, ts in phases.items():
    Xs, Y = [], {'size_next_dec': [], 'lead_t..t+3': [], 'ownc_t+1..t+3': []}
    cnt = {k: 0 for k in groups}
    for t in ts:
        sel = (fa >= t) & (fa < 24)
        Pw = (np.floor(K[:, None] * np.clip(THR[:, None] - c[:, :, t], 0, None)) * mine[:, :, t] * (v[:, :, t] > 0)).sum(1)
        T = (fa == t).astype(float)
        sz = size[:, t]
        cols = [np.ones(len(c))]
        for kk, f in groups.items():
            m = f(sz).astype(float); cols.append(T * Pw * m); cnt[kk] += int((sel & (T * Pw * m > 0)).sum())
        for kk, f in groups.items():
            m = f(sz).astype(float); cols += [Pw * m, m, T * m]
        cols += [own_c[:, t], oth_c[:, t], mg, sz] + ([own_c[:, t - 1]] if t > 0 else [])
        cols += [np.full(len(c), float(t == tt)) for tt in ts[1:]]
        Xs.append(np.column_stack(cols)[sel])
        nd = min(t - t % 4 + 4, 23) if t < 20 else 23
        Y['size_next_dec'].append(size[sel, nd])
        Y['lead_t..t+3'].append(((own_pool[:, t:t + 4] - oth_pool[:, t:t + 4]).sum(1) / 24)[sel])
        Y['ownc_t+1..t+3'].append(own_c[sel, t + 1:t + 4].sum(1) if t < 23 else 0)
    X = np.concatenate(Xs)
    print(ph, 'treated', cnt)
    for lab, y in Y.items():
        y = np.concatenate(y)
        XtXi = np.linalg.pinv(X.T @ X); bb = XtXi @ X.T @ y; r = y - X @ bb; se = np.sqrt(np.abs(np.diag(XtXi)) * (r ** 2).mean())
        print(f'   {lab:16s}', '  '.join(f'{kk}:{bb[1 + i]:+.4f}({se[1 + i]:.4f})' for i, kk in enumerate(groups)))
