import pandas as pd, numpy as np
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['ep']=d.episode
for run,g in d.groupby('run'):
    ne=g.ep.nunique()
    # group-level pool per round
    gr=g.groupby(['ep','round_number','group_id']).agg(pool=('common_good','first'),n=('contribution','size'),c=('contribution','mean'),p=('punishment','mean')).reset_index()
    full=gr.set_index(['ep','round_number','group_id']).unstack('group_id')
    pool=full['pool'].fillna(0).groupby('ep').mean()
    print(run,'eps',ne)
    print(' mean pool g0,g1',pool.mean().round(2).tolist(),' lead g0-g1',(pool[0]-pool[1]).mean().round(2),'se',((pool[0]-pool[1]).std()/np.sqrt(ne)).round(2))
    print(' members g0,g1',full['n'].fillna(0).mean().round(2).tolist(),' c',full['c'].mean().round(2).tolist(),' p',full['p'].mean().round(2).tolist())
    # by round
    br=gr.groupby(['round_number','group_id']).agg(n=('n','mean'),c=('c','mean'),p=('p','mean')).unstack('group_id')
    print(br.iloc[::2].round(2).to_string())
