"""v7: net score value of one cap-level late fine on a zero, by round r, and the event frequency.
value(r) = 0.15 * sum_{k>=1, r+k<=23} (own eff_k + 3 * others eff_k) - fine/2. v2 (S up to 16) + v4.
Frequency: under v6 (S=8), number of player-rounds with c=0 valid per round, and first-zero events after 8."""
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
for (lo, hi) in [(0, 0), (1, 2)]:
    print(f'c_r {lo}-{hi}')
    for rs in [(9, 10), (11, 12), (13, 14), (15, 16)]:
        val = []; n = [0, 0]; fines = []
        for r in range(rs[0], rs[1] + 1):
            e = {}
            for g, gm in [('ctl', S == r - 1), ('fin', S == r)]:
                gi, pi = np.where(gm[:, None] & (v[:, :, r] > 0) & (c[:, :, r] >= lo) & (c[:, :, r] <= hi))
                fcf = np.floor(A[gi] + Bb[gi] * (T[gi] - c[gi, pi, r])); m = fcf >= 24; gi, pi = gi[m], pi[m]
                own = np.nanmean(cc[gi, pi, r + 1:], 0)
                tot = np.nansum(cc[gi, :, r + 1:], 1) - np.nan_to_num(cc[gi, pi, r + 1:]); nn = v[gi, :, r + 1:].sum(1) - v[gi, pi, r + 1:]
                oth = np.nanmean(tot / np.maximum(nn, 1), 0)
                e[g] = (own, oth); n[g == 'fin'] += len(gi)
                if g == 'fin': fines.append(p[gi, pi, r].mean())
            gain = 0.15 * np.sum((e['fin'][0] - e['ctl'][0]) + 3 * (e['fin'][1] - e['ctl'][1]))
            gown = 0.15 * np.sum(e['fin'][0] - e['ctl'][0])
            val.append((gain, gown))
        f = np.mean(fines)
        print(f'  r {rs}: n ctl {n[0]} fin {n[1]}  fine {f:.1f}  gain own {np.mean([x[1] for x in val]):5.1f}  gain all {np.mean([x[0] for x in val]):5.1f}  net {np.mean([x[0] for x in val]) - f/2:+5.1f}')
z = np.load('/tmp/pf7/v6.npz', allow_pickle=True)
c6, v6 = z['c'].astype(float), z['v'].astype(float)
for F in [0, 1, 2]:
    low = (v6 > 0) & (c6 <= F)
    print(f'v6, c<={F}: per game, low player-rounds by round 9..23:', ' '.join(f'{x:.2f}' for x in low[:, :, 9:].sum(1).mean(0)))
    first = np.zeros(len(c6))
    print(f'   games with any c<={F} in rounds 9..15: {np.mean(low[:, :, 9:16].any((1, 2))):.3f}; player-rounds 9..15 per game {low[:, :, 9:16].sum((1, 2)).mean():.2f}')
