"""v3: cross-check the empirical surrogate (v2_surrogate.Surrogate) built on one cache against the
true scores of another sweep. usage: v3_surr_cv.py <cache.npz> <v1|v2> [step]"""
import numpy as np, json, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-8')
from v2_surrogate import Surrogate, v1_rule
S = Surrogate(sys.argv[1], sub=2 if "both" in sys.argv[1] else 1)
which = sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
def v2_rule(st, dose, cu, H):
    def r(Ch, Ph, t):
        if t >= H: return np.zeros(Ch.shape[:2])
        return dose * (np.clip(st - Ch[:, :, -1], 0, None) / st) ** cu
    return r
ver = 'pf-20261105-8' if which == 'v1' else 'pf-20261105-8-v2'
sw = json.load(open(f'plots/simulation/policy_finder/{ver}_sweep/sweep.json'))
true, est = [], []
for k, pt in enumerate(sw['points'][::step]):
    pr = pt['params']
    rule = v1_rule(pr['target'], pr['slope'], pr['horizon'], pr['cap']) if which == 'v1' else v2_rule(pr['standard'], pr['dose'], pr['curvature'], pr['horizon'])
    est.append(S.play(rule, n=1500, seed=k)[0]); true.append(pt['pool'])
true, est = np.array(true), np.array(est)
print(sys.argv[1], which, 'corr', np.corrcoef(true, est)[0, 1].round(3), 'bias', (est - true).mean().round(1))
top = np.argsort(-true)[:30]; print('corr top30', np.corrcoef(true[top], est[top])[0, 1].round(3))
print('zero', S.play(lambda Ch, Ph, t: np.zeros(Ch.shape[:2]), n=4000)[0], '(true 589.3)')
print('v1 final', S.play(v1_rule(13.75, 2.23, 12, 30), n=6000)[0], '(true 648.2)')
print('v2 final', S.play(v2_rule(19.28, 33.4, 2.23, 9), n=6000)[0], '(true 651.0)')
