"""v5: round-0 dose response in v4's sweep (part 1), group 0. c0 is set before any
punishment, so the round-0 dose varies only with the design point. For each c0 class,
regress the player's mean contribution in rounds 1-3, its stay at the round-3 decision and
its own net pool contribution over rounds 1-23 on the round-0 dose and the dose received in
rounds 1-3."""
import sys, numpy as np
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v4_sweep/per_round_p1of2.parquet')
c, p, g, v = A['c'], A['p'], A['g'] == 0, A['v']
m = g[:, 0] & v[:, 0]
c0 = c[:, 0][m]; d0 = p[:, 0][m]; d13 = p[:, 1:4].sum(1)[m]
y1 = c[:, 1:4].mean(1)[m]; stay = g[:, 4][m]
net = (1.6 * c[:, 1:] * g[:, 1:] - p[:, 1:] * g[:, 1:]).sum(1)[m]
for lo, hi in [(0, 2), (3, 5), (6, 8), (9, 11), (12, 20)]:
    s = (c0 >= lo) & (c0 <= hi)
    X = np.column_stack([np.ones(s.sum()), d0[s], d13[s]])
    out = []
    for y in [y1, stay.astype(float), net]:
        b, *_ = np.linalg.lstsq(X, y[s], rcond=None)
        out.append(b)
    print(f'c0 {lo}-{hi}: n={s.sum()} mean dose0={d0[s].mean():.1f} | c1-3: {out[0][0]:.2f} + {out[0][1]:.3f}/pt0 + {out[0][2]:.3f}/pt13'
          f' | stay: {out[1][0]:.3f} {out[1][1]*100:+.2f}pp/pt0 {out[1][2]*100:+.2f}pp/pt13 | net: {out[2][1]:+.2f}/pt0 {out[2][2]:+.2f}/pt13')
    # by dose bins
    for a, b_ in [(0, 0), (1, 9), (10, 19), (20, 29), (30, 30)]:
        q = s & (d0 >= a) & (d0 <= b_)
        if q.sum() > 500:
            print(f'    dose0 {a}-{b_}: n={q.sum()} c1-3={y1[q].mean():.2f} stay={stay[q].mean():.3f} net1-23={net[q].mean():.1f}')
