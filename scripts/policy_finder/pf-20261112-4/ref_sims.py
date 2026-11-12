import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['c']=d.contribution.where(d.contribution_valid.astype(bool),0)
for run,g in d.groupby('run'):
    gr=g.groupby(['episode','round_number','group_id']).agg(n=('c','size'),c=('c','mean'),p=('punishment','mean'),pool=('common_good','first')).reset_index()
    # pool per group: recompute
    pool=g.groupby(['episode','round_number','group_id']).apply(lambda x:1.6*x.c.sum()-x.punishment.sum()).rename('pool2').reset_index()
    gr=gr.merge(pool)
    full=gr.set_index(['episode','round_number','group_id']).pool2.unstack(fill_value=0)
    ep=full.groupby('episode').sum()/24
    lead=ep[0]-ep[1]
    print(run, 'pool g0 %.1f g1 %.1f lead %.2f se %.2f'%(ep[0].mean(),ep[1].mean(),lead.mean(),lead.std()/np.sqrt(len(lead))))
    s=gr.groupby(['group_id','round_number']).agg(n=('n','mean'),c=('c','mean'),p=('p','mean')).unstack(0)
    print(s.iloc[::2].round(2).to_string())
