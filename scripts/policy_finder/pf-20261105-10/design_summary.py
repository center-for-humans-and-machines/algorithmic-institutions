"""Summarise design_on_surrogate.py output: top points and score by param bins."""
import json, pandas as pd, numpy as np
d = pd.DataFrame([{**x['params'], 'score': x['score'], 'ptot': x['ptot']} for x in json.load(open('/tmp/pf10_design.json'))])
print(d.sort_values('score', ascending=False).head(15).round(2))
for col in ['k', 'target', 'base', 'horizon']:
    print(d.groupby(pd.qcut(d[col], 5), observed=True).score.agg(['mean', 'max']).round(1))
X = np.column_stack([np.ones(len(d)), np.log(d.k), d.target, d.base, np.log(d.horizon)])
print('score fraction above zero(586):', (d.score > 586).mean())
