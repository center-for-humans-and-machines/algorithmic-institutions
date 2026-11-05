"""v3: v2's surrogate (built on v1's sweep, or on another cache) scored out of sample on v2's sweep."""
import numpy as np, json, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from v2_surrogate import Surrogate
path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/v1sweep.npz'
S = Surrogate(path)
sw = json.load(open('plots/simulation/policy_finder/pf-20261105-8-v2_sweep/sweep.json'))
def v2_rule(st, dose, cu, H):
    def r(Ch, Ph, t):
        if t >= H: return np.zeros(Ch.shape[:2])
        return dose * (np.clip(st - Ch[:, :, -1], 0, None) / st) ** cu
    return r
true, est = [], []
for k, pt in enumerate(sw['points'][::2]):
    pr = pt['params']
    est.append(S.play(v2_rule(pr['standard'], pr['dose'], pr['curvature'], pr['horizon']), n=1500, seed=k)[0]); true.append(pt['pool'])
true, est = np.array(true), np.array(est)
print('corr', np.corrcoef(true, est)[0, 1].round(3), 'bias', (est - true).mean().round(1))
top = np.argsort(-true)[:30]; print('corr top30', np.corrcoef(true[top], est[top])[0, 1].round(3))
print('spearman', np.corrcoef(np.argsort(np.argsort(true)), np.argsort(np.argsort(est)))[0, 1].round(3))
print('zero', S.play(lambda Ch, Ph, t: np.zeros(Ch.shape[:2]), n=4000)[0])
