"""v3: on the surrogate built from both sweeps, vary only round 0 of v2's final rule."""
import numpy as np, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from v2_surrogate import Surrogate
S = Surrogate('/tmp/both.npz', sub=2)
def rule(st=19.28, dose=33.4, cu=2.23, H=9, r0=None):
    def r(Ch, Ph, t):
        if t >= H: return np.zeros(Ch.shape[:2])
        c = Ch[:, :, -1]
        base = dose * (np.clip(st - c, 0, None) / st) ** cu
        if t == 0 and r0 is not None: return r0(c, base)
        return base
    return r
variants = {
    'v2 final': None,
    'r0 none': lambda c, b: 0 * b,
    'r0 only c<8': lambda c, b: b * (c < 8),
    'r0 only c<5': lambda c, b: b * (c < 5),
    'r0 30 if c<5 else 0': lambda c, b: 30.0 * (c < 5),
    'r0 30 if c<8 else 0': lambda c, b: 30.0 * (c < 8),
    'r0 x1.5': lambda c, b: 1.5 * b,
    'r0 x0.5': lambda c, b: 0.5 * b,
    'r0 30*(c<8) + b*(c>=8)*.5': lambda c, b: 30.0 * (c < 8) + 0.5 * b * (c >= 8),
}
for name, f in variants.items():
    out = [S.play(rule(r0=f), n=6000, seed=s) for s in range(2)]
    print(f'{name:28s} {np.mean([o[0] for o in out]):7.1f}  c1-8 {out[0][2][1:9].mean():5.2f} c9-23 {out[0][2][9:].mean():5.2f} p0 {out[0][3][0]:5.2f}', flush=True)
