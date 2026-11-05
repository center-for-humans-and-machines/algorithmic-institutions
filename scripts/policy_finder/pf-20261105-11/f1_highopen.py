"""v6: groups whose round-0 lineup is all fairly high (min c0 in a bin). Does fining the lowest of them a little
at round 0 (because T is higher in that rule) pay? Outcome: group mean c by round, bill, episode score."""
import numpy as np
cs, ps, vs = [], [], []
for tag in ['v2', 'v4', 'v5']:
    z = np.load(f'/tmp/pf6/{tag}.npz', allow_pickle=True)
    cs.append(z['c']); ps.append(z['p']); vs.append(z['v'])
c = np.concatenate(cs).astype(float); p = np.concatenate(ps).astype(float); v = np.concatenate(vs).astype(float)
c = c * v
n = v.sum(1)
pay = 20 - c - p + ((1.6 * c.sum(1) - p.sum(1)) / np.maximum(n, 1))[:, None, :]
score = ((pay * v).sum(1) / np.maximum(n, 1) * (n > 0)).sum(1)
allv0 = v[:, :, 0].all(1)
mn = np.where(v[:, :, 0] > 0, c[:, :, 0], 99).min(1)
p0 = p[:, :, 0].sum(1)
cm = (c * v).sum(1) / np.maximum(n, 1)
for lo, hi in [(8, 11), (12, 14), (15, 20)]:
    m0 = allv0 & (mn >= lo) & (mn <= hi)
    print(f'min c0 {lo}-{hi}: n {m0.sum()}, mean c0 {cm[m0, 0].mean():.2f}')
    for a, b in [(0, 1), (1, 6), (6, 15), (15, 200)]:
        m = m0 & (p0 >= a) & (p0 < b)
        if m.sum() < 200: continue
        print(f'  p0tot [{a},{b}) n{m.sum():6d} c0 {cm[m,0].mean():5.2f} c1 {cm[m,1].mean():5.2f} c3 {cm[m,3].mean():5.2f} '
              f'c8 {cm[m,8].mean():5.2f} c15 {cm[m,15].mean():5.2f} c23 {cm[m,23].mean():5.2f} bill/pl {p[m].sum((1,2)).mean()/4:5.1f} score {score[m].mean():6.1f}')
