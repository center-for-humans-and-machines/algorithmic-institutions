# v7: at pos-1 rounds (1,5,9), does the response to punishment depend on the member's own drop from the previous round?
# my members valid at t-1,t, in my group at t-1..t+2; outcome: c at t+2 (decision round) - c_t ; leave = not in my group at t+3 (only for t=1,5,9 -> t+3=4,8,12)
import sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
rows = []
for tag in ['v3','v4','v5','v6']:
    d = load(f'/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-{tag}_sweep/', tag)
    c = d['c'].astype(np.float32); p = d['p'].astype(np.float32); m = d['mine']; v = d['v'].astype(bool)
    for t in [1,5,9]:
        ok = m[:,:,t-1] & m[:,:,t] & v[:,:,t] & v[:,:,t-1] & v[:,:,t+2] & m[:,:,t+2]
        okg = m[:,:,t] & v[:,:,t]
        tot = (c[:,:,t]*okg).sum(1, keepdims=True); cnt = okg.sum(1, keepdims=True)
        norm = (tot - c[:,:,t]) / np.maximum(cnt-1, 1)
        short = norm - c[:,:,t]
        drop = c[:,:,t-1] - c[:,:,t]
        dy = c[:,:,t+2] - c[:,:,t]
        leave = ~m[:,:,t+3]
        sel = ok & (cnt-1 >= 1)
        rows.append(np.stack([np.full(sel.sum(), t), short[sel], drop[sel], p[:,:,t][sel], dy[sel], leave[sel], c[:,:,t][sel], p[:,:,t-1][sel]], 1))
X = np.concatenate(rows)
t, s, dr, pp, dy, lv, cc, pprev = X.T
print('n', len(X))
for T in [1,5,9]:
    for lo, hi in [(1,4),(4,8),(8,20)]:
        sel = (t==T) & (s>=lo) & (s<hi)
        print(f't={T} short {lo}-{hi}')
        for dlo, dhi, lab in [(-20,-0.5,'rose'),(-0.5,0.5,'flat'),(0.5,3,'drop1-3'),(3,21,'drop3+')]:
            q = sel & (dr>dlo) & (dr<=dhi)
            if q.sum() < 500: continue
            A = np.column_stack([np.ones(q.sum()), pp[q], s[q], cc[q], pprev[q]])
            b = np.linalg.lstsq(A, dy[q], rcond=None)[0]
            bl = np.linalg.lstsq(A, lv[q], rcond=None)[0]
            print(f'   {lab:8s} n={q.sum():6d} mean dy={dy[q].mean():5.2f} dy/p={b[1]:.3f} leave={lv[q].mean():.3f} leave/p={bl[1]:.4f} meanp={pp[q].mean():.1f}')
