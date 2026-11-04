"""v2: did v1's pre_switch factor change leaving at the decisions among the harsher points?
Per point: our group's members after round-3 and round-7 decisions, regressed on params."""
import numpy as np
a = np.load('/tmp/v2_arrays.npz'); G = a['G']; prm = a['prm']; pool = a['pool']
n = (G == 0).sum(2).mean(1)  # [P, R] mean members per round
X = np.c_[np.ones(len(prm)), np.log(prm[:, 0]), prm[:, 1], prm[:, 2], np.minimum(prm[:, 3], 12), prm[:, 4]]
for r in [4, 8, 12, 16, 20]:
    b = np.linalg.lstsq(X, n[:, r], rcond=None)[0]
    print('members at round', r, 'coef [1, log slope, tol, cap, min(t_stop,12), pre_switch]:', b.round(3))
b = np.linalg.lstsq(X, pool, rcond=None)[0]; print('pool coef', b.round(2))
