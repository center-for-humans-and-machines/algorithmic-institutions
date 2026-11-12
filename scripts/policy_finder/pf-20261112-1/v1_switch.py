# v2: in v1's sweep (the rule's punishment varies across points, i.e. exogenously given contributions),
# how does a member of the rule's group respond, at a switch decision, to its own punishment
# (in the decision round vs earlier in the window) and to the punishment of the others?
import pandas as pd, numpy as np, glob
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
base='plots/simulation/policy_finder/pf-20261112-1_sweep/'
W=[]
import pyarrow.compute as pc
def chunks():
  for fi,f in enumerate(sorted(glob.glob(base+'per_round_p*.parquet'))):
    t=pq.read_table(f,columns=['episode','participant_code','round_number','punishment','contribution','agent_group','run'],read_dictionary=['run'])
    t=t.filter(pc.less_equal(t['round_number'],20))
    runs=t['run'].combine_chunks()
    cats=pd.Series(runs.dictionary.to_pylist())
    codes=runs.indices.to_numpy()
    for j in range(8):
        sel=(codes%8)==j
        d=t.filter(pa.array(sel)).to_pandas()
        c2=codes[sel]
        d['seat']=np.where(cats.str.split('by ').str[1].str.startswith('ah_vs').values[c2],1,0).astype(np.int8)
        d['run']=c2.astype(np.int32)+fi*100000
        yield d
import pyarrow as pa
for d in chunks():
    d=d.sort_values(['run','episode','participant_code','round_number'])
    k=['run','episode','participant_code']
    d['gnext']=d.groupby(k).agent_group.shift(-1)
    d['win']=d.round_number//4
    d['mine']=(d.agent_group==d.seat).astype(np.int8)
    d['isdec']=d.round_number%4==3
    # group-level per round
    gr=d.groupby(['run','episode','round_number','agent_group']).agg(gc=('contribution','sum'),gp=('punishment','sum'),gn=('contribution','size')).reset_index()
    d=d.merge(gr,on=['run','episode','round_number','agent_group'])
    o=gr.rename(columns={'gc':'oc','gp':'op','gn':'on'}); o['agent_group']=1-o.agent_group
    d=d.merge(o,on=['run','episode','round_number','agent_group'],how='left').fillna({'oc':0,'op':0,'on':0})
    d['pwin']=d.groupby(k+['win']).punishment.transform('sum')
    d['cwin']=d.groupby(k+['win']).contribution.transform('mean')
    d['gpwin']=d.groupby(k+['win']).gp.transform('sum')/d.groupby(k+['win']).gn.transform('sum')
    x=d[d.isdec & (d.round_number<20)].copy()
    x['sw']=(x.gnext!=x.agent_group).astype(float)
    x['pdec']=x.punishment; x['pearly']=x.pwin-x.punishment
    x['oth_pdec']=(x.gp-x.punishment)/(x.gn-1).clip(lower=1)
    x['share']=(1.6*x.gc-x.gp)/x.gn
    x['oshare']=np.where(x.on>0,(1.6*x.oc-x.op)/x.on.clip(lower=1),0)
    W.append(x[['run','episode','participant_code','round_number','win','mine','sw','pdec','pearly','oth_pdec','share','oshare','gn','on','contribution','cwin','gpwin','gc','gp']])
    del d
x=pd.concat(W)
x.to_parquet('/tmp/v1switch.parquet')
r=x[x.mine==1]
print('rule group members, decisions:',len(r))
m=smf.logit('sw ~ pdec + pearly + oth_pdec + share + oshare + gn + contribution + C(win)',r).fit(disp=0)
print(m.summary2().tables[1].round(4))
r=r.assign(cb=pd.cut(r.contribution,[-1,0,3,7,12,20]),pb=pd.cut(r.pdec,[-1,0,3,8,15,30]))
print(r.groupby(['cb','pb'],observed=True).sw.agg(['mean','size']).unstack().round(3))
a=x[x.mine==0]
m2=smf.logit('sw ~ pdec + pearly + oth_pdec + share + oshare + gn + contribution + C(win)',a).fit(disp=0)
print('ah group members'); print(m2.summary2().tables[1].round(4))
