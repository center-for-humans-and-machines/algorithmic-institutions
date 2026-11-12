"""v2: in the first-punishment-round experiment, do UNPUNISHED members (by exact contribution) of a group
whose other members were just punished change their next contribution, vs control groups? And per group:
total next-round contribution change, as a function of the group's total punishment that round."""
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
cbins = [(0, 4), (5, 9), (10, 14), (15, 20)]
gpb = [(1, 4), (5, 12), (13, 30), (31, 200)]
acc = {}
gacc = {}
for t in range(23):
    tr = fa == t; co = fa > t
    if tr.sum() == 0: continue
    m0 = mine[:, :, t] & (v[:, :, t] == 1) & (g[:, :, t + 1] == g[:, :, t]) & (v[:, :, t + 1] == 1)
    gp = (p[:, :, t] * mine[:, :, t]).sum(1)
    dc = c[:, :, t + 1] - c[:, :, t]
    for cv in range(21):
        cm = m0 & (c[:, :, t] == cv) & (p[:, :, t] == 0)
        C = cm & co[:, None]
        if C.sum() < 30: continue
        for i, (lo, hi) in enumerate(gpb):
            T = cm & tr[:, None] & (gp >= lo)[:, None] & (gp <= hi)[:, None]
            if T.sum() < 5: continue
            j = [k for k, (a, b) in enumerate(cbins) if a <= cv <= b][0]
            acc.setdefault((j, i), []).append((dc[T].mean() - dc[C].mean(), T.sum()))
print('UNPUNISHED members in groups just punished (group total bins) vs control, next-round dc')
for j, cb in enumerate(cbins):
    s = f'{str(cb):10s}'
    for i, gb in enumerate(gpb):
        a = np.array(acc.get((j, i), [(np.nan, 0)])); n = a[:, 1].sum()
        s += f'  grp{gb}: ' + (f'{np.average(a[:,0],weights=a[:,1]):+.2f} n{int(n)}' if n else '-')
    print(s)
