"""v3: on the surrogate built from both sweeps, v2's final rule plus a late tail: after the horizon
and until round H2, only players below a floor are punished (same convex dose, or a flat dose)."""
import numpy as np, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from v2_surrogate import Surrogate
S = Surrogate('/tmp/both.npz', sub=2)
def rule(st=19.28, dose=33.4, cu=2.23, H=9, floor=0, H2=9, flat=None):
    def r(Ch, Ph, t):
        c = Ch[:, :, -1]
        base = dose * (np.clip(st - c, 0, None) / st) ** cu
        if t < H: return base
        if t < H2: return (flat if flat is not None else base) * (c < floor)
        return 0 * c
    return r
for kw in [{}, dict(floor=3, H2=14), dict(floor=3, H2=18), dict(floor=5, H2=14), dict(floor=5, H2=18), dict(floor=8, H2=14),
           dict(floor=8, H2=18), dict(floor=5, H2=24), dict(floor=5, H2=18, flat=20.0), dict(H=7, floor=5, H2=16), dict(H=11, floor=5, H2=16)]:
    out = [S.play(rule(**kw), n=6000, seed=s) for s in range(2)]
    print(f'{str(kw):45s} {np.mean([o[0] for o in out]):7.1f} c9-23 {out[0][2][9:].mean():5.2f} p9-23 {out[0][3][9:].mean():5.2f}', flush=True)
