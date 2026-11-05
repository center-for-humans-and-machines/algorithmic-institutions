"""v4: in the sweeps' data, players at c<=3 in rounds 9-15: own and others' change
1 and 3 rounds later, by the dose they got (0, 1-9, 10-19, 20-30)."""
import sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v3_load
rows = []
for ver in ['pf-20261105-3', 'pf-20261105-3-v2', 'pf-20261105-3-v3']:
    a = v3_load.load(ver)
    c = a['c'] * a['v']; p = a['p']; v = a['v']
    for t in range(9, 16):
        m = (c[..., t] <= 3) & v[..., t] & v[..., t + 1] & v[..., t + 3]
        R, E, I = np.nonzero(m)
        for R_, E_, I_ in [(R, E, I)]:
            oth = [(np.arange(4) != i) for i in I_]
            oc = lambda k: np.array([c[r, e, o, k].mean() for r, e, o in zip(R_, E_, oth)])
            rows.append(np.stack([p[R_, E_, I_, t], c[R_, E_, I_, t], c[R_, E_, I_, t + 1] - c[R_, E_, I_, t],
                                  c[R_, E_, I_, t + 3] - c[R_, E_, I_, t],
                                  np.array([c[r, e, o, t + 3].mean() - c[r, e, o, t].mean() for r, e, o in zip(R_, E_, oth)]),
                                  np.array([c[r, e, o, t].mean() for r, e, o in zip(R_, E_, oth)])], 1))
x = np.concatenate(rows)
print('n', len(x))
for lo, hi in [(0, 0), (1, 9), (10, 19), (20, 30)]:
    m = (x[:, 0] >= lo) & (x[:, 0] <= hi)
    for olo, ohi in [(0, 12), (12, 21)]:
        mm = m & (x[:, 5] >= olo) & (x[:, 5] < ohi)
        print(f'p {lo}-{hi} others {olo}-{ohi}: n {mm.sum():6d} own +1 {x[mm,2].mean():+.2f} own +3 {x[mm,3].mean():+.2f} others +3 {x[mm,4].mean():+.2f}')
