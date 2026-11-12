# v6: (a) what-if on v5 points with bar >= 11: keep a first push only if gap >= g and target mean <= b;
# (b) size-6 one-target pushes by finer gap and decision (v2/v3 rows), with the dose and later punishment.
import numpy as np, pandas as pd
T=pd.read_parquet('/tmp/v6_all_touched.parquet'); P=pd.read_parquet('/tmp/v6_v5_paired.parquet')
N=(P.bar>=11).sum(); s=T[(T.v=='v5')&(T.p_bar>=11)]
print('N games',N,'v5 gain/game on these points %.3f'%(s.dlead.sum()/N))
res=pd.DataFrame({g:{b:s.dlead[(s.gap>=g)&(s.tw<=b)].sum()/N for b in [2,4,6,8,10,12]} for g in [0,2,3,4,5,6]}); print(res.round(3))
o=T[(T.ntg==1)&(T['size']==6)&(T.t0%4==3)&(T.dose>=10)]
print(o.pivot_table(index=['t0'],columns=pd.cut(o.gap,[-30,4,6,8,10,12,30]),values='dlead',aggfunc=lambda x:'%.2f(%d)'%(x.mean(),len(x)),observed=True))
print(o.groupby('v').size())
o7=o[o.t0==7]
print(o7.pivot_table(index=pd.cut(o7.later,[-1,0,30,1000]),columns=pd.cut(o7.gap,[-30,6,8,30]),values='dlead',aggfunc=lambda x:'%.2f(%d)'%(x.mean(),len(x)),observed=True))
