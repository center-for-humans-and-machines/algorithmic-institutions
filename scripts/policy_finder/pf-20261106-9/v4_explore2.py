# v4: in v3's sweep, does the effect of our punishment on a member's later contribution depend on the
# punishment it received before, from anyone (us or the other manager)? OLS on our members who stay
# in our group from t to t+h, controls: own c[t], c[t-1], group mean at t.
import numpy as np
d = np.load('/tmp/v3sweep.npz')
c, p, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
pu = p * mine
cum_all = np.cumsum(p, -1) - p          # punishment from anyone before round t
cum_us = np.cumsum(pu, -1) - pu
def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    return np.linalg.lstsq(X, y, rcond=None)[0]
for t in [1, 4, 5, 8, 9, 12, 16]:
    for h in [1, 3]:
        sel = mine[:, :, t] & mine[:, :, t + h]
        for r in range(t, t + h):
            sel &= mine[:, :, r]
        gm = (c[:, :, t] * mine[:, :, t]).sum(1) / np.maximum(mine[:, :, t].sum(1), 1)
        gm = np.broadcast_to(gm[:, None], c.shape[:2])
        y = c[:, :, t + h][sel]
        pt_ = pu[:, :, t][sel]
        e_us = np.log1p(cum_us[:, :, t][sel]); e_oth = np.log1p((cum_all - cum_us)[:, :, t][sel])
        X = [c[:, :, t][sel], c[:, :, t - 1][sel], gm[sel], pt_, pt_ * e_us, pt_ * e_oth, e_us, e_oth]
        b = ols(X, y)
        print(f't={t:2d} h={h} n={sel.sum():7d} b_p={b[4]:.3f} p*log1p(prior us)={b[5]:+.4f} p*log1p(prior other)={b[6]:+.4f} '
              f'| mean log1p us {e_us.mean():.2f} oth {e_oth.mean():.2f} | b_us={b[7]:+.2f} b_oth={b[8]:+.2f}')
