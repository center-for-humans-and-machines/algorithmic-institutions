"""v4: heterogeneity of the round-4 effect by the state of the two groups at round 4: own minus other contribution
pool (1.6 * sum c, before punishment) and own size. Treatment as in v4_bs_replicate.py (total P4 minus its design
mean given history). All points, and last_b == 1 points only (clean rest-of-game)."""
import numpy as np, json
D = np.load('/tmp/v3sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
base = 'plots/simulation/policy_finder/pf-20261112-6-v3_sweep/'
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = np.array(sorted(pts))
PAR = np.array([[pts[k]['n_min'], pts[k]['last_b'], pts[k]['thr'], pts[k]['streak'], pts[k]['amt']] for k in keys])
pid = np.searchsorted(keys, name)
N = len(c); b = 4
mine = g == mg[:, None, None]
op = (1.6 * c * mine - p * mine).sum(1); oo = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1)
lead = (op[:, b:] - oo[:, b:]).sum(1) / 24
lead_early = (op[:, b:8] - oo[:, b:8]).sum(1) / 24
def would(par):
    nmin, lb, thr, st, amt = par
    low = np.ones((N, 8), bool)
    for k in range(int(st)): low &= c[:, :, b - k] <= thr
    here = mine[:, :, b] & (v[:, :, b] > 0)
    return (np.floor(amt) * (low & here).sum(1) * (size[:, b] >= nmin)).astype(np.float32)
W = np.stack([would(PAR[i]) for i in range(len(keys))], 1)
P = W[np.arange(N), pid]
gap = 1.6 * ((c * mine)[:, :, b].sum(1) - (c * ~mine)[:, :, b].sum(1))
gap_pc = (c * mine)[:, :, b].sum(1) / np.maximum(size[:, b], 1) - (c * ~mine)[:, :, b].sum(1) / np.maximum(8 - size[:, b], 1)
print('gap quantiles', np.percentile(gap, [10, 25, 50, 75, 90]))
def fit(y, R, E, sel):
    X = np.column_stack([np.ones(sel.sum()), R[sel], E[sel]])
    bb, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
    e = y[sel] - X @ bb
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e ** 2).mean())
    return f'{bb[1]:+.4f}({se[1]:.4f})'
for lab, lbsel in [('all points', np.ones(len(keys), bool)), ('last_b==1', PAR[:, 1] == 1)]:
    E = W[:, lbsel].mean(1); R = P - E; ok = lbsel[pid]
    print(lab)
    for gl, s in [('gap<0', gap < 0), ('gap 0-30', (gap >= 0) & (gap < 30)), ('gap 30-60', (gap >= 30) & (gap < 60)), ('gap>=60', gap >= 60)]:
        sel = ok & s
        if R[sel].std() < 1e-6: continue
        print(f'  {gl:10s} games {sel.sum():6d} lead {fit(lead, R, E, sel)} lead r4-7 {fit(lead_early, R, E, sel)}'
              f' size r8 {fit(size[:, 8].astype(float), R, E, sel)} size r20 {fit(size[:, 20].astype(float), R, E, sel)}')
    for gl, s in [('pc<-2', gap_pc < -2), ('pc -2..2', np.abs(gap_pc) <= 2), ('pc>2', gap_pc > 2)]:
        sel = ok & s
        if R[sel].std() < 1e-6: continue
        print(f'  {gl:10s} games {sel.sum():6d} lead {fit(lead, R, E, sel)} size r8 {fit(size[:, 8].astype(float), R, E, sel)}')
