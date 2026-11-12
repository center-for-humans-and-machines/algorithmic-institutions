import pandas as pd, sys
d=pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d['pair']=d.run.str.split('by ').str[1]
# group pool per round
g=d.groupby(['pair','episode','round_number','group_id']).agg(cg=('common_good','first'),n=('participant_code','size'),c=('contribution','mean'),p=('punishment','mean')).reset_index()
# fill missing groups as 0
for pair,gp in g.groupby('pair'):
    piv=gp.pivot_table(index=['episode','round_number'],columns='group_id',values='cg').fillna(0)
    n=gp.pivot_table(index=['episode','round_number'],columns='group_id',values='n').fillna(0)
    ep=piv.groupby('episode').mean()
    diff=ep[0]-ep[1]
    print(pair, 'eps',len(ep),'pool0 %.1f pool1 %.1f diff %.1f se %.1f'%(ep[0].mean(),ep[1].mean(),diff.mean(),diff.std()/len(ep)**.5), 'n0 %.2f n1 %.2f'%(n[0].mean(),n[1].mean()))
    print(gp.groupby(['group_id']).agg(c=('c','mean'),p=('p','mean')).round(2).to_dict())
    print(n.groupby('round_number').mean().round(2).T.to_string())
