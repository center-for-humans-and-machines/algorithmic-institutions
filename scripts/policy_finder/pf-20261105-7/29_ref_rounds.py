"""Per-round mean contribution (valid only) and valid share under zero/humanlike, and v2 final best."""
import pandas as pd, numpy as np
df=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet',columns=['run','round_number','contribution','punishment','contribution_valid'])
for r,g in df.groupby('run'):
    v=g[g.contribution_valid]
    print(r, ' '.join(f'{x:.1f}' for x in v.groupby('round_number').contribution.mean().values))
    print('  valid', ' '.join(f'{x:.2f}' for x in g.groupby('round_number').contribution_valid.mean().values))
a=np.load('/tmp/pf7/pf-20261105-7-v2_final.npz'); m=a['R']=='pf-20261105-7-v2_s132'
C=a['C'][m].astype(float);V=a['V'][m].astype(float)
print('v2 s132', ' '.join(f'{x:.1f}' for x in (C*V).sum((0,1))/V.sum((0,1))))
