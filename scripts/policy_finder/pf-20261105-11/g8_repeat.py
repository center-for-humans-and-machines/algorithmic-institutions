"""v7: value of fining a zero again who did not respond to a cap fine last round.
c_r = 0, c_{r-1} = 0 and p_{r-1} >= 24 (so S >= r-1); S = r (fined again) vs S = r-1 (left alone). r = 6..14."""
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
for rs in [(6, 9), (10, 14)]:
    vals = []
    for r in range(rs[0], rs[1] + 1):
        e = {}; nn_ = {}
        for g, gm in [('ctl', S == r - 1), ('fin', S == r)]:
            gi, pi = np.where(gm[:, None] & (v[:, :, r] > 0) & (c[:, :, r] == 0) & (c[:, :, r - 1] == 0) & (p[:, :, r - 1] >= 24))
            fcf = np.floor(A[gi] + Bb[gi] * (T[gi] - c[gi, pi, r])); m = fcf >= 24; gi, pi = gi[m], pi[m]
            own = np.nanmean(cc[gi, pi, r + 1:], 0)
            tot = np.nansum(cc[gi, :, r + 1:], 1) - np.nan_to_num(cc[gi, pi, r + 1:]); nn = v[gi, :, r + 1:].sum(1) - v[gi, pi, r + 1:]
            e[g] = (own, np.nanmean(tot / np.maximum(nn, 1), 0)); nn_[g] = len(gi)
        go = 0.15 * np.sum(e['fin'][0] - e['ctl'][0]); ga = go + 0.45 * np.sum(e['fin'][1] - e['ctl'][1])
        vals.append((go, ga, nn_['ctl'], nn_['fin'], (e['fin'][0] - e['ctl'][0])[0]))
    a = np.array(vals)
    print(f'r {rs}: n ctl {a[:,2].sum():.0f} fin {a[:,3].sum():.0f}  own lag1 eff {a[:,4].mean():+.1f}  gain own {a[:,0].mean():.1f} all {a[:,1].mean():.1f}  net {a[:,1].mean()-14.5:+.1f}')
