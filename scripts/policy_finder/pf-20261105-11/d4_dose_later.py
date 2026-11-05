"""v4: dose-response of c_{t+1} to own fine p_t in rounds 2-9, at fixed own c_t and c_{t-1}
(regression with c_t x c_{t-1} cell means removed), by c_t band."""
import numpy as np
cs, ps, vs = [], [], []
for tag in ['v1', 'v2', 'v3']:
    z = np.load(f'/tmp/pf4/{tag}.npz', allow_pickle=True)
    cs.append(z['c']); ps.append(z['p']); vs.append(z['v'])
c = np.concatenate(cs).astype(int); p = np.concatenate(ps).astype(float); v = np.concatenate(vs)
for trange in [(1, 3), (4, 8), (9, 14)]:
    rows = []
    for t in range(*trange):
        ok = v[:, :, t] & v[:, :, t + 1] & v[:, :, t - 1] & v[:, :, t].all(1)[:, None]
        oc = (c[:, :, t].sum(1, keepdims=True) - c[:, :, t]) / 3
        op = (p[:, :, t].sum(1, keepdims=True) - p[:, :, t]) / 3
        rows.append(np.column_stack([c[:, :, t][ok], c[:, :, t - 1][ok], p[:, :, t - 1][ok], oc[ok], p[:, :, t][ok], op[ok], c[:, :, t + 1][ok]]))
    R = np.concatenate(rows)
    ct, cp, pp, oc, pt, op, cn = R.T
    # residualise on cell (ct, cp) and others' c (rounded)
    key = ct * 1000 + cp * 40 + np.minimum(pp, 30).astype(int) // 5 * 1 + 0
    key = key * 100 + np.round(oc).astype(int)
    _, inv = np.unique(key, return_inverse=True)
    mean_cn = np.bincount(inv, cn) / np.bincount(inv)
    print(f'rounds {trange[0]}-{trange[1]-1}: N {len(R)}')
    pb = [0, 1, 2, 4, 6, 9, 13, 18, 31]
    for lo, hi in [(0, 4), (5, 8), (9, 11), (12, 13), (14, 15), (16, 20)]:
        m = (ct >= lo) & (ct <= hi)
        out = []
        for a, b in zip(pb[:-1], pb[1:]):
            mm = m & (pt >= a) & (pt < b)
            # within-cell deviation of c_{t+1} vs cell mean, shows the p effect net of state
            out.append(f'[{a},{b}):{(cn[mm]-mean_cn[inv[mm]]).mean():+5.2f}/{mm.sum():6d}' if mm.sum() > 300 else ' ' * 20)
        print(f'  c_t {lo:2d}-{hi:2d}', ' '.join(out))
