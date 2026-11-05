"""v2: across v1's 256 rules, does punishing in the late rounds pay? Per run: payoff of
rounds 12-23 vs mean punishment in rounds 12-19, controlling for the state at round 11."""
import numpy as np, pandas as pd, json
from v2_load import load
a = load(); c, p, v = a['c'], a['p'], a['v']
n = v.sum(2)  # [run, ep, round]
pay = np.where(n > 0, 20 + (0.6 * (c * v).sum(2) - 2 * (p * v).sum(2)) / np.maximum(n, 1), 0)  # mean valid payoff
cm = (c * v).sum(2) / np.maximum(n, 1); pm = (p * v).sum(2) / np.maximum(n, 1)
R = pd.DataFrame({'score': pay.sum(-1).mean(1), 'pay_late': pay[..., 12:].sum(-1).mean(1), 'pay_early': pay[..., :12].sum(-1).mean(1),
                  'c11': cm[..., 9:12].mean((1, 2)), 'c23': cm[..., 21:].mean((1, 2)), 'p_late': pm[..., 12:20].mean((1, 2)),
                  'p_early': pm[..., :3].mean((1, 2)), 'p_mid': pm[..., 3:12].mean((1, 2))})
s = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261105-3_sweep/sweep.json'))
print('check score vs sweep.json', R.score.iloc[:3].round(1).tolist(), [round(x['pool'], 1) for x in s['points'][:3]])
good = R[R.c11 > 15]
print(len(good), 'runs with c at rounds 9-11 > 15')
X = np.column_stack([np.ones(len(good)), good.c11, good.p_late])
b = np.linalg.lstsq(X, good.pay_late.values, rcond=None)[0]
print('pay rounds 12-23 ~ c11, p_late:', b.round(2))
print(good.sort_values('p_late')[['score', 'pay_early', 'pay_late', 'c11', 'c23', 'p_early', 'p_mid', 'p_late']].round(2).to_string())
