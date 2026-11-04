"""v3 sweep: per round 0-7, mean group levels (c, p, cg, n) for both groups at the v3 final point's
neighbours (best-scoring sweep points) and for all points; how often my pre-punishment cg is behind."""
import json, numpy as np, pandas as pd
pd.set_option('display.width', 250)
W = pd.read_parquet('/tmp/v3_wide.parquet')
s = json.load(open('plots/simulation/policy_finder/pf-20261104-7-v3_sweep/sweep.json'))
pts = pd.DataFrame([{**p['params'], 'pool': p['pool'], 'run': int(p['name'].split('_s')[-1])} for p in s['points']])
top = pts.sort_values('pool', ascending=False).head(15)
print(top.round(2).to_string())
T = W[W.run.isin(top.run) & (W.round_number < 12)]
T = T[(T.n0 > 0)]
T['pre0'] = T.cg0 + T.p0 * T.n0 / T.n0  # approx: cg0 + mean punishment (valid ~ all)
T['behind_pre'] = (T.pre0 < T.cg1)
T['behind_post'] = (T.cg0 < T.cg1)
print(T.groupby('round_number')[['n0','c0','c1','p0','p1','cg0','cg1','pre0','behind_pre','behind_post']].mean().round(2).to_string())
