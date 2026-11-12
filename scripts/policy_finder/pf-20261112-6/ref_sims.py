import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
# per group/round pool and size
g=d.groupby(['run','episode','round_number','group_id']).agg(pool=('common_good','first'),n=('contribution','size'),c=('contribution','mean'),p=('punishment','mean')).reset_index()
for run,gr in g.groupby('run'):
    print(run)
    print(gr.groupby('group_id')[['pool','n','c','p']].mean().round(2))
    # full episode avg pool per group, empty group counts 0
    ep=gr.pivot_table(index=['episode','round_number'],columns='group_id',values='pool').fillna(0)
    m=ep.groupby('episode').mean(); diff=m[0]-m[1]
    print('lead g0-g1 %.2f se %.2f'%(diff.mean(),diff.std()/np.sqrt(len(diff))))
    print(gr.pivot_table(index='round_number',columns='group_id',values='n',aggfunc='sum').div(gr.episode.nunique()).round(2).iloc[::4].T)
