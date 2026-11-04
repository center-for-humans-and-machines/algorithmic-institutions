"""Static cost of v2 candidates on contributions seen under v1's final gp point (no feedback), vs v1's actual spending."""
import numpy as np
import pandas as pd
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261104-7_sweep/final/per_round.parquet',
                    columns=['run', 'round_number', 'contribution', 'punishment', 'agent_group', 'contribution_valid'])
d = d[d.run.str.contains('_gp_') & (d.agent_group == 0)]
print('v1 gp actual: mean p/member-round %.2f, share punished %.2f' % (d.punishment.mean(), (d.punishment > 0).mean()))
def v2(thr, base, rate, te, taper):
    c, t = d.contribution.values, d.round_number.values
    tw = np.clip(1 - taper * t / te, 0, None) * (t < te)
    p = (base + rate * np.clip(thr - c, 0, None)) * (c < thr) * tw * d.contribution_valid.values
    return np.floor(np.clip(p, 0, 30))
for args in [(12, 0, 2.29, 11, 0), (12, 3, 1.0, 14, 0.7), (14, 2, 0.8, 16, 1), (9, 2, 0.5, 12, 0.5), (16, 6, 3, 20, 1), (6, 0, 0.2, 6, 1)]:
    p = v2(*args)
    print(args, 'mean p/member-round %.2f, share punished %.2f, mean p when punished %.1f' % (p.mean(), (p > 0).mean(), p[p > 0].mean() if (p > 0).any() else 0))
