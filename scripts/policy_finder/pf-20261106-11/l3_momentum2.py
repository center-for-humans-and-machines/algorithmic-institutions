# v7: momentum check, extended: block starts (t=4,8) and pos-1 rounds, outcome both at the block's decision round (t+k to 4b+3)
# and at the next block's decision round (4b+7) for members who stay through; slopes from a regression with p x drop-bin
import sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
rows = []
for tag in ['v3','v4','v5','v6']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c = d['c'].astype(np.float32); p = d['p'].astype(np.float32); m = d['mine']; v = d['v'].astype(bool)
    for t in [1,4,5,8]:
        b = t//4; end = 4*b+3; end2 = end+4
        ok = m[:,:,t] & v[:,:,t] & v[:,:,t-1] & v[:,:,end] & m[:,:,end]
        okg = m[:,:,t] & v[:,:,t]
        tot = (c[:,:,t]*okg).sum(1, keepdims=True); cnt = okg.sum(1, keepdims=True)
        norm = (tot - c[:,:,t]) / np.maximum(cnt-1, 1)
        short = norm - c[:,:,t]; drop = c[:,:,t-1] - c[:,:,t]
        stay2 = m[:,:,end2] & v[:,:,end2]
        sel = ok & (cnt-1 >= 1) & (short >= 1)
        rows.append(np.stack([np.full(sel.sum(), t), short[sel], drop[sel], p[:,:,t][sel], (c[:,:,end]-c[:,:,t])[sel], (~m[:,:,end+1])[sel], c[:,:,t][sel], stay2[sel], (c[:,:,end2]-c[:,:,t])[sel]], 1))
X = np.concatenate(rows)
t, s, dr, pp, dy, lv, cc, st2, dy2 = X.T
bins = [(-20,-0.5,'rose'),(-0.5,0.5,'flat'),(0.5,3,'drop1-3'),(3,21,'drop3+')]
for T in [1,4,5,8]:
    q = t==T
    cols = [np.ones(q.sum()), s[q], cc[q]]
    names = []
    for lo,hi,lab in bins:
        ind = ((dr[q]>lo)&(dr[q]<=hi)).astype(float); cols += [ind*pp[q]]; names.append(lab)
        if lab!='rose': cols += [ind]
    A = np.column_stack(cols)
    def coef(y, w=None):
        if w is None: return np.linalg.lstsq(A, y, rcond=None)[0]
        return np.linalg.lstsq(A[w], y[w], rcond=None)[0]
    idx = [3,4,6,8]  # positions of p x bin coefficients
    b1 = coef(dy[q]); bl = coef(lv[q]); w = st2[q]>0; b2 = coef(dy2[q], w)
    print(f't={T} n={q.sum()}')
    for k, lab in zip(idx, names):
        share = ((dr[q]>bins[names.index(lab)][0])&(dr[q]<=bins[names.index(lab)][1])).mean()
        print(f'   {lab:8s} share={share:.2f}  dy_end/p={b1[k]:.3f}  dy_next_end/p={b2[k]:.3f}  leave/p={bl[k]:.4f}')
