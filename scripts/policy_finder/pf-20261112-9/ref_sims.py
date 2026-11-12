import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['cv']=d.contribution_valid.astype(float)
for run,g in d.groupby('run'):
    print('==',run, 'episodes',g.episode.nunique())
    # group pool per round
    gp=g.groupby(['episode','round_number','group_id']).agg(pool=('common_good','first'),n=('participant_code','size'),c=('contribution','mean'),p=('punishment','mean')).reset_index()
    full=gp.pivot_table(index=['episode','round_number'],columns='group_id',values=['pool','n','c','p'])
    # pool averaged with empty=0
    pv=gp.pivot_table(index=['episode','round_number'],columns='group_id',values='pool').fillna(0)
    m=pv.groupby('episode').mean()
    print('mean pool g0,g1',m.mean().round(2).tolist(),'diff',(m[0]-m[1]).mean().round(2),'se',((m[0]-m[1]).std()/np.sqrt(len(m))).round(2))
    print(gp.groupby(['group_id']).agg(c=('c','mean'),p=('p','mean'),n=('n','mean')).round(2))
    s=gp.groupby(['round_number','group_id']).agg(n=('n','mean'),c=('c','mean'),p=('p','mean')).unstack()
    print(s.iloc[::2].round(2))
