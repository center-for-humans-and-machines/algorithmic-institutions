import sys; sys.path.insert(0,'scripts/policy_finder/pf-20261106-3')
from load import load
import numpy as np
df = load(); c = df.contribution.values
for thr in [4,8,10,12,16]:
    s = np.clip(thr-c,0,None)
    row = [f'{np.floor(np.clip(k*s,0,30)).mean():.2f}' for k in [0.05,0.1,0.2,0.4,0.8,2.0]]
    print(thr, 'mean shortfall', round(s.mean(),2), 'mean punishment at slope .05,.1,.2,.4,.8,2:', row)
