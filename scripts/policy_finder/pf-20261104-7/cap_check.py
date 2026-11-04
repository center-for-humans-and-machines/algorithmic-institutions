"""Static check of a common-good gap cap on v3's 15 best sweep points: per round 0-7, the share of
group-rounds where 'my post-punishment cg >= other cg + margin' would bind, and the mean punishment
per member it would remove (using the other group's actual cg as the estimate; no feedback)."""
import json, numpy as np, pandas as pd
pd.set_option('display.width', 250)
W = pd.read_parquet('/tmp/v3_wide.parquet')
s = json.load(open('plots/simulation/policy_finder/pf-20261104-7-v3_sweep/sweep.json'))
pts = pd.DataFrame([{'pool': p['pool'], 'run': int(p['name'].split('_s')[-1])} for p in s['points']])
top = pts.sort_values('pool', ascending=False).head(15)
T = W[W.run.isin(top.run) & (W.round_number < 8) & (W.n0 > 0)].copy()
T['pre0'] = T.cg0 + T.p0
for m in [-10, -5, -2, 0, 3]:
    budget = np.clip(T.pre0 - T.cg1 - m, 0, None)
    cut = np.clip(T.p0 - budget, 0, None)
    T[f'bind{m}'] = (cut > 0).astype(float)
    T[f'cut{m}'] = cut
print(T.groupby('round_number')[['p0'] + [f'bind{m}' for m in [-10, -5, -2, 0, 3]] + [f'cut{m}' for m in [-10, -5, -2, 0, 3]]].mean().round(2).to_string())
