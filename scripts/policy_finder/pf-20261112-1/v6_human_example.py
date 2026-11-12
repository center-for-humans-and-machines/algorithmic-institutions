# v6: human data. At decision rounds 7/11/15, groups of >= 5: was the lowest window-mean member punished hard (>= 10)
# in the decision round? Change of the others' (who stay) mean contribution from this window to the next, by gap.
import pandas as pd, numpy as np
d=pd.read_csv('/opt/algorithmic-institutions/experiments/2group_8agent_50ep.csv').drop_duplicates(['session','episode','round_number','participant_code'])
d['win']=d.round_number//4
w=d.groupby(['session','episode','participant_code','win']).agg(c=('contribution','mean'),g=('group_id','last')).reset_index()
last=d[d.round_number%4==3][['session','episode','participant_code','win','group_id','punishment','experiment_name']]
x=last.merge(w,on=['session','episode','participant_code','win'])
nx=w.copy(); nx['win']-=1; x=x.merge(nx.rename(columns={'c':'c_next','g':'g_next'}),on=['session','episode','participant_code','win'],how='left')
rows=[]
for (s,e,wi,g),grp in x[x.win.isin([1,2,3])].groupby(['session','episode','win','group_id']):
    if len(grp)<5: continue
    j=grp.c.idxmin(); o=grp.drop(j); st=o[o.g_next==g]
    if len(st)<2: continue
    rows.append(dict(exp=grp.experiment_name.iloc[0],n=len(grp),gap=o.c.mean()-grp.c[j],hard=grp.punishment[j]>=10,ow=o.c.mean(),
                     d_oth=(st.c_next-st.c).mean(),tleft=grp.g_next[j]!=g,oleft=(o.g_next!=g).mean(),pun_oth=o.punishment.mean()))
R=pd.DataFrame(rows); print(len(R)); print(R.groupby('exp').size())
print(R.pivot_table(index=pd.cut(R.gap,[-1,4,8,20]),columns='hard',values=['d_oth','oleft','pun_oth'],aggfunc='mean',observed=True).round(2))
print(R.pivot_table(index=pd.cut(R.gap,[-1,4,8,20]),columns='hard',values='d_oth',aggfunc='size',observed=True))
