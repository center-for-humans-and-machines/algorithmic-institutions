"""v7: late dose-response. v2+v4 sweeps, rounds 9..15, players with c_r in a low bin, in games with S = r (one fine
at r, then silence) vs S = r-1 (no fine). Fine size at r binned (it varies with T, A, B). Own c at r+1, mean of
r+1..r+6; others' mean c r+1..r+6 (others may also be fined at r)."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
def params(tag):
    d = json.load(open(f'{B}pf-20261105-11-{tag}_sweep/sweep.json'))
    return {pt['name']: pt['params'] for pt in d['points']}
C, Pp, V, SS = [], [], [], []
for tag in ['v2', 'v4']:
    z = np.load(f'/tmp/pf7/{tag}.npz', allow_pickle=True)
    P = params(tag)
    C.append(z['c'].astype(float)); Pp.append(z['p'].astype(float)); V.append(z['v'].astype(float))
    SS.append(np.array([P[r.split(' managed by ')[-1]]['S'] for r in z['run']]))
c, p, v, S = map(np.concatenate, (C, Pp, V, SS))
cc = np.where(v > 0, c, np.nan)
for lo, hi in [(0, 0), (1, 5), (6, 9), (10, 12)]:
    print(f'=== c_r {lo}-{hi}')
    for rr in [(9, 11), (12, 15)]:
        acc = {}
        for r in range(rr[0], rr[1] + 1):
            for g in ['ctl', 'fin']:
                gm = (S == r - 1) if g == 'ctl' else (S == r)
                gi, pi = np.where(gm[:, None] & (v[:, :, r] > 0) & (c[:, :, r] >= lo) & (c[:, :, r] <= hi))
                f = p[gi, pi, r]
                o1 = cc[gi, pi, r + 1]; o6 = np.nanmean(cc[gi, pi, r + 1:r + 7], 1)
                oth = (np.nansum(cc[gi, :, r + 1:r + 7], (1, 2)) - np.nansum(cc[gi, pi, r + 1:r + 7], 1)) / np.maximum(1, (v[gi, :, r + 1:r + 7].sum((1, 2)) - v[gi, pi, r + 1:r + 7].sum(1)))
                othf = p[gi, :, r].sum(1) - f
                for fb in ([(0, 0)] if g == 'ctl' else [(1, 5), (6, 10), (11, 16), (17, 22), (23, 30)]):
                    m = (f >= fb[0]) & (f <= fb[1])
                    acc.setdefault((g, fb), []).append((o1[m], o6[m], oth[m], othf[m], f[m]))
        print(f' rounds {rr}')
        base = None
        for k, lst in acc.items():
            o1 = np.concatenate([x[0] for x in lst]); o6 = np.concatenate([x[1] for x in lst]); oth = np.concatenate([x[2] for x in lst]); of = np.concatenate([x[3] for x in lst]); f = np.concatenate([x[4] for x in lst])
            if len(o1) < 40: continue
            print(f'  {k[0]} fine {k[1]}: n {len(o1):5d} fine {f.mean():4.1f} others fined {of.mean():4.1f}  own r+1 {np.nanmean(o1):4.1f}  own r+1..6 {np.nanmean(o6):4.1f}  others r+1..6 {np.nanmean(oth):4.1f}')
