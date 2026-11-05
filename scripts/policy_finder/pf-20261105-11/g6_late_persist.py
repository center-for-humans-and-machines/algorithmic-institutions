"""v7: persistence of a late cap-level fine on a zero contributor. v2+v4, r = 9..13, c_r = 0 (and 1-3),
counterfactual fine >= 24: S = r (fined at r only) vs S = r-1. Effect on own c per lag 1..10, on others' c,
split by the player's history (c at round 8 >= 10: a recent collapse; <= 5: a chronic low)."""
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
for (lo, hi) in [(0, 0), (1, 3)]:
  for hist in ['all', 'recent', 'chronic']:
    own = {'ctl': [], 'fin': []}; oth = {'ctl': [], 'fin': []}; of = {'fin': [], 'ctl': []}; zero_again = {'ctl': [], 'fin': []}
    for r in range(9, 14):
        for g, gm in [('ctl', S == r - 1), ('fin', S == r)]:
            gi, pi = np.where(gm[:, None] & (v[:, :, r] > 0) & (c[:, :, r] >= lo) & (c[:, :, r] <= hi))
            fcf = np.floor(A[gi] + Bb[gi] * (T[gi] - c[gi, pi, r]))
            h = c[gi, pi, r - 3]
            m = fcf >= 24
            if hist == 'recent': m &= h >= 10
            if hist == 'chronic': m &= h <= 5
            gi, pi = gi[m], pi[m]
            L = np.full((len(gi), 10), np.nan); O = np.full((len(gi), 10), np.nan)
            for k in range(1, 11):
                if r + k > 23: break
                L[:, k - 1] = cc[gi, pi, r + k]
                tot = np.nansum(cc[gi, :, r + k], 1) - np.nan_to_num(cc[gi, pi, r + k]); nn = v[gi, :, r + k].sum(1) - v[gi, pi, r + k]
                O[:, k - 1] = tot / np.maximum(nn, 1)
            own[g].append(L); oth[g].append(O); of[g].append(p[gi, :, r].sum(1) - p[gi, pi, r])
            zero_again[g].append(np.nanmean(np.where(np.isnan(L[:, 1:6]), np.nan, L[:, 1:6] <= 2), 1))
    Lc, Lf = np.concatenate(own['ctl']), np.concatenate(own['fin']); Oc, Of = np.concatenate(oth['ctl']), np.concatenate(oth['fin'])
    print(f'c_r {lo}-{hi} {hist}: n ctl {len(Lc)} fin {len(Lf)}  others fined at r {np.concatenate(of["fin"]).mean():.1f}  share at c<=2 in lags 2-6: ctl {np.nanmean(np.concatenate(zero_again["ctl"])):.2f} fin {np.nanmean(np.concatenate(zero_again["fin"])):.2f}')
    print('   own ctl  ', ' '.join(f'{x:5.1f}' for x in np.nanmean(Lc, 0)))
    print('   own eff  ', ' '.join(f'{x:+5.1f}' for x in np.nanmean(Lf, 0) - np.nanmean(Lc, 0)))
    print('   oth eff  ', ' '.join(f'{x:+5.2f}' for x in np.nanmean(Of, 0) - np.nanmean(Oc, 0)))
