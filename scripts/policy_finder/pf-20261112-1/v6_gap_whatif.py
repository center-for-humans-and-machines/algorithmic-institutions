# v6: v5's touched games by the others' window mean (ow) and target's (tw) at the first push; what-if gain per game
# (over all 72634 paired v5 games) if only first pushes meeting a condition were kept (later pushes not re-simulated).
import numpy as np, pandas as pd
T=pd.read_parquet('/tmp/v6_v5_touched.parquet'); N=72634; T['gap']=T.ow-T.tw
print(T.pivot_table(index=pd.cut(T.ow,[0,4,6,8,10,12,20]),columns=pd.cut(T.tw,[-1,0,2,4,6,12]),values='dlead',aggfunc='mean',observed=True).round(2))
print(T.pivot_table(index=pd.cut(T.ow,[0,4,6,8,10,12,20]),columns=pd.cut(T.tw,[-1,0,2,4,6,12]),values='dlead',aggfunc='size',observed=True))
print(T.pivot_table(index=pd.cut(T.gap,[-20,2,4,6,8,10,20]),columns='t0',values='dlead',aggfunc='mean',observed=True).round(2))
print(T.pivot_table(index=pd.cut(T.ow,[0,4,6,8,10,12,20]),columns='t0',values='dlead',aggfunc='mean',observed=True).round(2))
for g in [0,2,3,4,5,6,7,8]:
    m=T.gap>=g; print('gap>=%d touched %.3f gain/game %.3f  per touched %.2f'%(g,m.sum()/N,T.dlead[m].sum()/N,T.dlead[m].mean()))
for o in [0,5,6,7,8,9,10]:
    m=T.ow>=o; print('ow>=%d touched %.3f gain/game %.3f'%(o,m.sum()/N,T.dlead[m].sum()/N))
for f in [1,1.5,2,2.5,3]:
    m=T.ow>=f*T.tw+3; print('ow>=%.1f*tw+3 gain/game %.3f'%(f,T.dlead[m].sum()/N))
# per-bar: is the gap condition interacting with the bar
for b in [5,8,12]:
    s=T[T.bar==b]; n=(T.bar==b).sum()
    print('bar',b,'all %.2f gap>=5 %.2f'%(s.dlead.mean(),s.dlead[s.gap>=5].mean()))
