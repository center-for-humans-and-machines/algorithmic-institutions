"""v2: careful comparison (several seeds, large n) of a few candidates on the surrogate."""
import numpy as np
from v2_surrogate import Surrogate
S = Surrogate()
cur = lambda Ch: Ch[:, :, -1]
def lin(T, s, H): return lambda Ch, Ph, t: (t < H) * s * np.clip(T - cur(Ch), 0, None)
def pw(T, M, H, g): return lambda Ch, Ph, t: (t < H) * M * (np.clip(T - cur(Ch), 0, None) / T) ** g
cands = {'zero': lambda Ch, Ph, t: 0 * cur(Ch), 'v1 final 13.75/2.23/12': lin(13.75, 2.23, 12), 'lin 15.5/1.6/8': lin(15.5, 1.6, 8),
         'pow 19/30/10/2': pw(19, 30, 10, 2), 'pow 17/30/10/1.5': pw(17, 30, 10, 1.5), 'pow 21/30/10/2.5': pw(21, 30, 10, 2.5),
         'pow 19/30/12/2': pw(19, 30, 12, 2), 'pow 15/30/10/1.5': pw(15, 30, 10, 1.5), 'pow 21/30/10/1.5': pw(21, 30, 10, 1.5)}
for name, r in cands.items():
    sc = [S.play(r, n=8000, seed=s)[0] for s in (11, 12, 13)]
    _, _, c, p = S.play(r, n=4000, seed=1)
    print(f'{name:26s} {np.mean(sc):6.1f} (seeds {np.round(sc, 1)}) c@[0,4,9,15,23] {np.round(c[[0, 4, 9, 15, 23]], 1)} sum p {p.sum():.1f}', flush=True)
