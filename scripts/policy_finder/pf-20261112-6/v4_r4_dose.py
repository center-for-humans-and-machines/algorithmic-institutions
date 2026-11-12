"""v4: dose and target of round-4 punishment on v3's sweep. Same design as v4_bs_replicate.py (treatment = would-be
punishment minus its mean over the design given the history). (a) restricted to last_b == 1 points, which punish at
round 4 only, so the rest-of-game outcome is that of round 4 alone; (b) split by the target's round-4 contribution;
(c) split by the point's amt (per-member dose)."""
import numpy as np, json, os, sys
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
own_c = (c * mine)[:, :, b + 1:].sum(1).mean(1)
def would_m(par):
    nmin, lb, thr, st, amt = par
    low = np.ones((N, 8), bool)
    for k in range(int(st)): low &= c[:, :, b - k] <= thr
    here = mine[:, :, b] & (v[:, :, b] > 0)
    return (np.floor(amt) * (low & here) * (size[:, b] >= nmin)[:, None]).astype(np.float32)  # [N,8]
# per game, would-be punishment per target category per point: [N,256,3] (by c4) and actual P
c4_ = c[:, :, b]
CAT = np.stack([c4_ <= 1, (c4_ >= 2) & (c4_ <= 4), c4_ >= 5], -1).astype(np.float32)  # [N,8,3]
W = np.zeros((N, len(keys), 3), np.float32)
for i in range(len(keys)):
    W[:, i] = np.einsum('nj,njk->nk', would_m(PAR[i]), CAT)
def fit(y, Rs, Es, sel, labs):
    X = np.column_stack([np.ones(sel.sum())] + [r[sel] for r in Rs] + [e[sel] for e in Es])
    bb, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
    e = y[sel] - X @ bb
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e ** 2).mean())
    return '  '.join(f'{l} {bb[i+1]:+.4f}({se[i+1]:.4f})' for i, l in enumerate(labs))
P = W[np.arange(N), pid]  # [N,3] actual by category
c4 = c[:, :, b]
# (a) last_b == 1 only
sel = PAR[pid, 1] == 1
E = W[:, PAR[:, 1] == 1].sum(2).mean(1)
print('(a) last_b==1 points, games', sel.sum())
print('  lead', fit(lead, [P.sum(1) - E], [E], sel, ['P4']), '| own c', fit(own_c, [P.sum(1) - E], [E], sel, ['P4']))
for lab, s2 in [('n<=6', size[:, b] <= 6), ('n>=7', size[:, b] >= 7)]:
    print('   ', lab, 'lead', fit(lead, [P.sum(1) - E], [E], sel & s2, ['P4']))
# (b) by target contribution (all points)
sel = np.ones(N, bool)
cats = [('c<=1', 0), ('c2-4', 1), ('c5+', 2)]
Rs, Es = [], []
for l, m in cats:
    Pk = P[:, m]; Ek = W[:, :, m].mean(1)
    Rs.append(Pk - Ek); Es.append(Ek)
print('(b) by target c4: lead', fit(lead, Rs, Es, sel, [l for l, _ in cats]))
print('                  own c', fit(own_c, Rs, Es, sel, [l for l, _ in cats]))
# (c) by per-member dose
bins = [(2, 4), (4, 7), (7, 11), (11, 17)]
Rs, Es = [], []
for lo, hi in bins:
    msk = (PAR[:, 4] >= lo) & (PAR[:, 4] < hi)
    Pk = P.sum(1) * msk[pid]; Ek = (W.sum(2) * msk[None]).mean(1)
    Rs.append(Pk - Ek); Es.append(Ek)
print('(c) by amt: lead', fit(lead, Rs, Es, sel, [f'amt{lo}-{hi}' for lo, hi in bins]))
print('            own c', fit(own_c, Rs, Es, sel, [f'amt{lo}-{hi}' for lo, hi in bins]))
# (d) by streak
Rs, Es = [], []
for s in [1, 2, 3]:
    msk = PAR[:, 3] == s
    Pk = P.sum(1) * msk[pid]; Ek = (W.sum(2) * msk[None]).mean(1)
    Rs.append(Pk - Ek); Es.append(Ek)
print('(d) by streak: lead', fit(lead, Rs, Es, sel, ['st1', 'st2', 'st3']))
