"""Static spending of v3 candidates per round block on the contributions seen under v2's final gp
(no feedback), next to v2 gp's actual spending: mean punishment per member-round in rounds 0-3, 4-11, 12-23."""
import numpy as np
import pandas as pd
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261104-7-v2_sweep/final/per_round.parquet',
                    columns=['run', 'round_number', 'contribution', 'punishment', 'agent_group', 'contribution_valid'])
d = d[d.run.str.contains('v2_gp_') & (d.agent_group == 0)]
blk = np.select([d.round_number < 4, d.round_number < 12], ['0-3', '4-11'], '12-23')
print('v2 gp actual', d.groupby(blk).punishment.mean().round(2).to_dict())
def v3(thr, rate, early, te, shape):
    c, t = d.contribution.values, d.round_number.values.astype(float)
    size = np.minimum(rate * np.clip(thr - c, 0, None), 30)
    fade = np.clip(1 - (t - 4) / (te - 4), 0, 1) ** shape
    w = np.where(t < 4, early, np.where(t < te, fade, 0))
    return np.floor(np.clip(size * w * d.contribution_valid.values, 0, 30))
for args in [(9.7, 6, 1, 8, 1), (9.7, 6, 0, 12, 1), (10, 10, 0.3, 12, 1), (10, 3, 0.5, 16, 2), (8, 1.5, 0, 6, 0), (14, 30, 1, 20, 0)]:
    p = pd.Series(v3(*args), index=d.index)
    print(args, p.groupby(blk).mean().round(2).to_dict(), 'share punished %.2f' % (p > 0).mean())
