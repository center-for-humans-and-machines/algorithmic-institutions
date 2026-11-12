"""Human data: one copy per game; contribution response of low contributors to punishment; net membership vs block punishment."""
import pandas as pd, numpy as np
d=pd.read_csv('experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
print(len(d), d.groupby(['session','episode']).ngroups,'games')
d=d.sort_values(['session','episode','participant_code','round_number'])
k=['session','episode','participant_code']
d['c_next']=d.groupby(k).contribution.shift(-1); d['g_next']=d.groupby(k).group_id.shift(-1)
d['c_prev']=d.groupby(k).contribution.shift(1)
x=d[(d.g_next==d.group_id)&(d.player_no_input==0)&d.c_next.notna()].copy()
x['dc']=x.c_next-x.contribution
x['pb']=pd.cut(x.punishment,[-1,0,3,8,30]); x['cb']=pd.cut(x.contribution,[-1,0,5,10,20])
print(x.pivot_table(index='cb',columns='pb',values='dc').round(2)); print(x.pivot_table(index='cb',columns='pb',values='dc',aggfunc='size'))
print('human mean punishment per round:', d.groupby('round_number').punishment.mean().round(2).values)
# block-level: group's mean punishment over block vs net change in members at the decision
g=d.groupby(['session','episode','round_number','group_id']).agg(n=('contribution','size'),p=('punishment','mean'),c=('contribution','mean')).reset_index()
rows=[]
for (s,e),ge in g.groupby(['session','episode']):
    piv=ge.pivot_table(index='round_number',columns='group_id',values=['n','p','c']).fillna(0)
    for r in [3,7,11,15,19]:
        if r+1 not in piv.index: continue
        blk=piv.loc[r-3:r]
        rows.append(dict(r=r,dn0=piv.loc[r+1,('n',0)]-piv.loc[r,('n',0)],pd=blk[('p',0)].mean()-blk[('p',1)].mean(),cd=blk[('c',0)].mean()-blk[('c',1)].mean()))
b=pd.DataFrame(rows)
X=np.column_stack([np.ones(len(b)),b.pd,b.cd]); print('net members gained by g0 ~ const + pun diff + contr diff:',np.linalg.lstsq(X,b.dn0,rcond=None)[0].round(3), len(b))
