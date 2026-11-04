"""Reference sims: per-group pool, size, contribution by run and phase."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
print(d[['agent_group','group_id']].drop_duplicates().head())
d['phase'] = d.round_number // 4
for run, g in d.groupby('run'):
    print('==', run)
    gr = g.groupby(['episode','round_number','group_id']).agg(
        pool=('common_good','first'), n=('contribution','size'),
        c=('contribution','mean'), p=('punishment','mean'),
        csum=('contribution','sum'), psum=('punishment','sum')).reset_index()
    gr['pool2'] = 1.6*gr.csum - gr.psum
    full = gr.set_index(['episode','round_number','group_id']).unstack('group_id').fillna(0)
    for gid in (0,1):
        print(' group', gid, 'pool/round (empty=0): %.2f' % full[('pool2',gid)].mean(),
              'common_good col %.2f' % full[('pool',gid)].mean(),
              'mean size %.2f' % full[('n',gid)].mean())
    gr['phase']=gr.round_number//4
    print(gr.groupby(['group_id','phase'])[['n','c','p']].mean().unstack('group_id').round(2))
