"""v7: g4 matched. For controls (S = r-1) compute the fine the rule's params would have charged at r
(floor(A + B*(T-c))), so fined (S = r) and unfined players are compared within the same param region.
Effect = fin - ctl on own c (r+1, r+1..r+6) and on others' c r+1..r+6. Payoff accounting per fined low player."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
def params(tag):
    d = json.load(open(f'{B}pf-20261105-11-{tag}_sweep/sweep.json'))
    return {pt['name']: pt['params'] for pt in d['points']}
C, Pp, V, PR = [], [], [], []
for tag in ['v2', 'v4']:
    z = np.load(f'/tmp/pf7/{tag}.npz', allow_pickle=True)
    P = params(tag)
    C.append(z['c'].astype(float)); Pp.append(z['p'].astype(float)); V.append(z['v'].astype(float))
    PR.append(np.array([[P[r.split(' managed by ')[-1]][k] for k in 'TABS'] for r in z['run']]))
c, p, v, PR = map(np.concatenate, (C, Pp, V, PR))
T, A, Bb, S = PR.T
cc = np.where(v > 0, c, np.nan)
def table(lo, hi, rr):
    out = {}
    for r in range(rr[0], rr[1] + 1):
        for g, gm in [('ctl', S == r - 1), ('fin', S == r)]:
            gi, pi = np.where(gm[:, None] & (v[:, :, r] > 0) & (c[:, :, r] >= lo) & (c[:, :, r] <= hi))
            fcf = np.clip(np.floor(np.where(c[gi, pi, r] < T[gi], A[gi] + Bb[gi] * (T[gi] - c[gi, pi, r]), 0)), 0, 30)
            o1 = cc[gi, pi, r + 1]; o6 = np.nanmean(cc[gi, pi, r + 1:r + 7], 1)
            oth = (np.nansum(cc[gi, :, r + 1:r + 7], (1, 2)) - np.nansum(cc[gi, pi, r + 1:r + 7], 1)) / np.maximum(1, (v[gi, :, r + 1:r + 7].sum((1, 2)) - v[gi, pi, r + 1:r + 7].sum(1)))
            othc_r = (np.nansum(cc[gi, :, r], 1) - c[gi, pi, r]) / np.maximum(1, v[gi, :, r].sum(1) - 1)
            for fb in [(1, 8), (9, 14), (15, 20), (21, 26), (27, 30)]:
                m = (fcf >= fb[0]) & (fcf <= fb[1])
                out.setdefault(fb, {}).setdefault(g, []).append((o1[m], o6[m], oth[m], othc_r[m], fcf[m]))
    print(f' c_r {lo}-{hi}, rounds {rr}')
    for fb, d in out.items():
        res = {}
        for g in ['ctl', 'fin']:
            arr = [np.concatenate([x[i] for x in d[g]]) for i in range(5)]
            res[g] = arr
        n0, n1 = len(res['ctl'][0]), len(res['fin'][0])
        if min(n0, n1) < 40: continue
        e = [np.nanmean(res['fin'][i]) - np.nanmean(res['ctl'][i]) for i in range(4)]
        se6 = np.sqrt(np.nanvar(res['fin'][1]) / n1 + np.nanvar(res['ctl'][1]) / n0)
        print(f'  fine {fb}: n ctl {n0:5d} fin {n1:5d}  mean fine {res["fin"][4].mean():4.1f} | others c_r ctl {np.nanmean(res["ctl"][3]):4.1f} fin {np.nanmean(res["fin"][3]):4.1f} | own r+1 {e[0]:+4.1f}  own r+1..6 {e[1]:+4.1f} (se {se6:.2f})  others r+1..6 {e[2]:+4.2f}')
for lo, hi in [(0, 0), (1, 5), (6, 9), (10, 12)]:
    for rr in [(9, 11), (12, 15)]:
        table(lo, hi, rr)
