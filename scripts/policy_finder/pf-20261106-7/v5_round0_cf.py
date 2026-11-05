# v5: counterfactual round-0 allocation on the simulated round-0 contributions (exogenous to any rule).
# target_i = (1-w)*c0 + w*others_mean_i; p_i = min(30, floor(k*relu(target_i - c_i))). Predicted gain from
# v5_ah_gap.py (t+8 lift per sqrt p = 0.716 + 0.038*(gap-2.42); exit per sqrt p = 0.0364 - 0.0013*(gap-2.42))
# and v5_ah_spill.py (+0.048 per point dealt to others, for each other member)
import numpy as np
d = np.load('/tmp/v4sweep.npz'); C = d['C'][:, :, 0].astype(float); G = d['G'][:, :, 0]; V = d['V'][:, :, 0] > 0; seat = d['seat']
mine = (G == seat[:, None]) & V
s = np.where(mine, C, 0).sum(1, keepdims=True); n = mine.sum(1, keepdims=True)
om = (s - C) / np.maximum(n - 1, 1); gap = om - C
def run(c0, k, w):
    tgt = (1 - w) * c0 + w * om
    p = np.where(mine, np.minimum(30, np.floor(k * np.maximum(0, tgt - C))), 0)
    sq = np.sqrt(p)
    direct = ((0.716 + 0.038 * (gap - 2.42)) * sq * mine).sum(1)
    spill = 0.048 * (p.sum(1, keepdims=True) - p) * mine
    exits = ((0.0364 - 0.0013 * (gap - 2.42)) * sq * mine).sum(1)
    return p.sum(1).mean() / 4, direct.mean(), spill.sum(1).mean(), exits.mean(), (p > 0).sum(1).mean()
print('c0 k w | p/member  direct  spill  exits  n_punished')
for c0, k in [(9, 11.4), (10, 6.9)]:
    base = run(c0, k, 0)
    print(c0, k, 0, np.round(base, 3))
    for w in [0.25, 0.5, 0.75, 1.0]:
        # match the mean punishment by scaling k (bisection)
        lo, hi = 0.5, 60
        for _ in range(40):
            km = (lo + hi) / 2
            if run(c0, km, w)[0] < base[0]: lo = km
            else: hi = km
        for c0b in [c0 - 2, c0, c0 + 2]:
            pass
        print(c0, round(km, 2), w, np.round(run(c0, km, w), 3))
print('round-0 others mean: mean %.2f sd %.2f; own c sd %.2f' % (om[mine].mean(), om[mine].std(), C[mine].std()))
