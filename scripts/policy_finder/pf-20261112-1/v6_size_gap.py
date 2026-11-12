# v6: does a cooperative group (gap/ow) make pushes pay where v5 forbids them: sizes <= 6, decision 19 (and 3), several targets.
import numpy as np, pandas as pd
T=pd.read_parquet('/tmp/v6_all_touched.parquet')
T['gb']=pd.cut(T.gap,[-30,4,6,8,30]); T['dec']=(T.t0%4==3)
def tab(s,idx,title):
    print('--',title); print(s.pivot_table(index=idx,columns='gb',values='dlead',aggfunc=lambda x:'%.2f(%d)'%(x.mean(),len(x)),observed=True))
one=T[(T.ntg==1)&T.dec&(T.dose>=13)]
tab(one,['size'],'one target, dose>=13, decision rounds, all versions, by size')
tab(one[one['size']>=7],['t0'],'one target, size>=7, by decision')
tab(one[one['size']<=6],['v','t0'],'one target size<=6 by version and decision')
tab(T[T.dec&(T.dose>=13)&(T['size']>=7)],['ntg'],'size>=7 by number of targets')
tab(T[T.dec&(T.dose>=13)&(T['size']<=6)],['ntg'],'size<=6 by number of targets')
print(one.groupby('v').later.describe())
