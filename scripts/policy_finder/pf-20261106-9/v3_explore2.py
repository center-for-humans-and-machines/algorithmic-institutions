# v3: in v2's sweep, does a punishment's effect on later contribution depend on whether the member
# was punished by us before (first vs repeat), and on the round? OLS on our members who stay in
# our group from t-1 to t+h within the block.
import numpy as np
d = np.load('/tmp/v2sweep.npz')
c, p, g, ours = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours']
mine = (g == ours[:, None, None])
pu = p * mine
cum = np.cumsum(pu, -1) - pu          # punishment from us before round t
cnt = np.cumsum(pu > 0, -1) - (pu > 0)  # number of earlier punished rounds
def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b
for t in [0, 1, 4, 5, 8, 9, 12]:
    for h in [1, 3]:
        if (t % 4) + h > 3 and t % 4 == 0 and h == 3:
            pass
        stay = mine[:, :, t] & mine[:, :, t + h] & mine[:, :, t - 1 if t > 0 else 0]
        if (t // 4) != ((t + h) // 4):
            stay &= mine[:, :, (t // 4) * 4 + 3]
        gm = (c[:, :, t] * mine[:, :, t]).sum(1) / np.maximum(mine[:, :, t].sum(1), 1)
        gm = np.broadcast_to(gm[:, None], c.shape[:2])
        sel = stay
        y = c[:, :, t + h][sel]
        pt_ = pu[:, :, t][sel]
        prior = (cnt[:, :, t][sel] > 0).astype(np.float32)
        X = [c[:, :, t][sel], (c[:, :, t - 1] if t > 0 else c[:, :, t])[sel], gm[sel], pt_, pt_ * prior, prior]
        b = ols(X, y)
        print(f't={t:2d} h={h} n={sel.sum():8d} prior-share={prior.mean():.2f}  b_p(first)={b[4]:.3f}  b_p*prior={b[5]:+.3f}  b_prior={b[6]:+.2f}')
