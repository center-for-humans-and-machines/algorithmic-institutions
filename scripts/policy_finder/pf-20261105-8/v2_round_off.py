"""v2: in the surrogate, v1's best rule with the punishment of one round switched off,
or doubled: the marginal value of each round's punishment."""
import numpy as np
from v2_surrogate import Surrogate
S = Surrogate()
def rule(off=-1, mult=1.0, T=13.75, s=2.23, H=12):
    return lambda Ch, Ph, t: (t < H) * (mult if t == off else 1.0) * s * np.clip(T - Ch[:, :, -1], 0, None)
base = S.play(rule(), n=12000, seed=3)[0]
print('base', round(base, 1))
for t in [0, 1, 2, 3, 5, 8, 11]:
    a = S.play(rule(off=t, mult=0), n=12000, seed=3)[0]; b = S.play(rule(off=t, mult=0.5), n=12000, seed=3)[0]
    print(t, 'off', round(a - base, 1), 'half', round(b - base, 1))
