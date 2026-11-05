# v2: characterise the opponents and my v1 rule's play in the final run of v1
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv)>1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11_sweep/final/per_round.parquet'
pt = sys.argv[2] if len(sys.argv)>2 else 'pf-20261106-11_gp'
d = pd.read_parquet(f)
d = d[d.run.str.contains(pt+'_vs_')|d.run.str.contains('_vs_'+pt+'$')]
d['mine_g0'] = d.run.str.contains('by '+pt+'_vs_')
d['opp'] = d.run.str.extract(r'(opponent_\d)')[0]
d['mine'] = np.where(d.mine_g0, d.group_id.astype(str)==d.group_id.astype(str).min(), d.group_id.astype(str)!=d.group_id.astype(str).min())
print(d.group_id.unique(), d.agent_group.unique()[:5])
d['cb'] = pd.cut(d.contribution,[-1,0,5,10,15,19,20])
for o,g in d.groupby('opp'):
    print('=====',o)
    gr = g.groupby(['mine','run','episode','round_number']).agg(n=('contribution','size'),c=('contribution','mean'),p=('punishment','mean'),pool=('common_good','first')).reset_index()
    print(gr.groupby(['mine', pd.cut(gr.round_number,[-1,3,7,11,15,19,23])])[['n','c','p']].mean().unstack(0).round(2))
    print('punish by contribution bin (mine, opp):')
    print(g.groupby(['mine','cb'],observed=True).punishment.mean().unstack(0).round(2))
