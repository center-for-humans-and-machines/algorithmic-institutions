"""Reference sims (ah_vs_ah, zero side of zero_vs_ah): group mean seen by low contributors; leave-one-out mean."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
for run in ['managed by ah_vs_ah', 'managed by zero_vs_ah']:
    x = d[d.run.str.contains(run) & (d.group_id==0) & (d.contribution_valid==1)].copy()
    k = x.groupby(['episode','round_number']).contribution
    x['gm'] = k.transform('mean'); x['n'] = k.transform('size')
    x['loo'] = ((x.gm*x.n - x.contribution)/(x.n-1).clip(lower=1)).where(x.n>1, x.contribution)
    lo = x[x.contribution<=4]
    print(run, 'share c<=4: %.2f' % (x.contribution<=4).mean())
    print('  group mean seen by c<=4:', lo.gm.quantile([.1,.25,.5,.75,.9]).round(1).tolist(), ' leave-one-out:', lo.loo.quantile([.1,.25,.5,.75,.9]).round(1).tolist())
    print('  group mean by round block', x.groupby(x.round_number//4).gm.mean().round(2).tolist())
