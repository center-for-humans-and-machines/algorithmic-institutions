"""v7: what does a late fine do to a collapsed (low) player? Natural experiment from the S draws in v2/v4 sweeps:
players with c_r <= F at round r (9..14), in games with S = r-1 (no fine at r, nor later) vs S >= r (fined at r,
and later while S lasts). Outcomes: own c at r+1, r+3, r+6, others' c, and the fine paid.
Also: unfined natural recovery in v6 (S=8)."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
def params(tag):
    nm = 'pf-20261105-11-' + tag
    d = json.load(open(f'{B}{nm}_sweep/sweep.json'))
    return {pt['name']: pt['params'] for pt in d['points']}
rows = []
for tag in ['v2', 'v4']:
    z = np.load(f'/tmp/pf7/{tag}.npz', allow_pickle=True)
    c, p, v, run = z['c'].astype(float), z['p'].astype(float), z['v'].astype(float), z['run']
    P = params(tag)
    S = np.array([P[r.split(' managed by ')[-1]]['S'] for r in run])
    T = np.array([P[r.split(' managed by ')[-1]]['T'] for r in run])
    rows.append((c, p, v, S, T))
c = np.concatenate([r[0] for r in rows]); p = np.concatenate([r[1] for r in rows]); v = np.concatenate([r[2] for r in rows])
S = np.concatenate([r[3] for r in rows]); T = np.concatenate([r[4] for r in rows])
cc = np.where(v > 0, c, np.nan)
for F in [5, 9]:
    print(f'=== players with c_r <= {F}')
    for r in [9, 10, 11, 12, 13, 14]:
        res = {}
        for lab, gm in [('stop r-1', S == r - 1), ('fined r ', S >= r), ('fined r only (S=r)', S == r)]:
            gi, pi = np.where(gm[:, None] & (v[:, :, r] > 0) & (c[:, :, r] <= F))
            if len(gi) < 30: continue
            out = [np.nanmean(cc[gi, pi, min(r + k, 23)]) for k in [0, 1, 3, 6]]
            oth = np.nanmean([np.nanmean(np.delete(cc[g, :, r + 1:r + 7], q, 0)) for g, q in zip(gi[:3000], pi[:3000])])
            fine = p[gi, pi, r].mean(); bill = p[gi, :, r:].sum((1, 2)).mean()
            print(f' r {r:2d} {lab:18s} n {len(gi):5d}  c_r {out[0]:4.1f} r+1 {out[1]:4.1f} r+3 {out[2]:4.1f} r+6 {out[3]:4.1f}  others r+1..6 {oth:4.1f}  fine_r {fine:4.1f} group bill r.. {bill:5.1f}')
z = np.load('/tmp/pf7/v6.npz', allow_pickle=True)
c6, v6 = z['c'].astype(float), z['v'].astype(float); cc6 = np.where(v6 > 0, c6, np.nan)
print('=== v6 (no fines after 8): low players at round r, natural path')
for r in [10, 12, 15]:
    for lo, hi in [(0, 0), (1, 5), (6, 9)]:
        gi, pi = np.where((v6[:, :, r] > 0) & (c6[:, :, r] >= lo) & (c6[:, :, r] <= hi))
        prev = np.nanmean(cc6[gi, pi, r - 1])
        print(f' r {r} c {lo}-{hi}: n {len(gi)} c_(r-1) {prev:4.1f} r+1 {np.nanmean(cc6[gi,pi,r+1]):4.1f} r+3 {np.nanmean(cc6[gi,pi,min(r+3,23)]):4.1f} r+6 {np.nanmean(cc6[gi,pi,min(r+6,23)]):4.1f}')
