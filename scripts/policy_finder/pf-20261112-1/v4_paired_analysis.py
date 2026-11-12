# v4: breakdowns of the paired effects written by v4_paired_sweeps.py
import pandas as pd, numpy as np
pd.set_option('display.width',220); pd.set_option('display.max_rows',200)
df=pd.read_parquet('/tmp/v4_paired.parquet')
def agg(g): return g.agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem'),down=('down','mean'),doth=('doth','mean'),dmem=('dmem','mean'),left=('left','mean'),total=('total','mean')).round(2)
v3=df[df.v=='v3']
for k in ['t0','size','ntg','p_n_push','p_amount','p_bar','p_last_dec','nrounds']:
    print(agg(v3.groupby(k)))
v3=v3.assign(tb=pd.cut(v3.tmean,[-.1,0,0.5,1,2,3,4,6]), db=pd.cut(v3.dose,[0,10,15,20,25,30]))
print(agg(v3.groupby('tb',observed=True)))
print(agg(v3.groupby('db',observed=True)))
print(agg(v3.groupby(['t0','size'])))
print(agg(v3.groupby(['t0','ntg'])))
print('==== v2')
v2=df[df.v=='v2']
print(agg(v2.groupby(['t0','size'])))
print(agg(v2.groupby(['size','ntg'])))
print('==== v2 single push games (post-treatment selection!)')
q=v2[(v2.nrounds==1)&(v2.ntg==1)&(v2.t0<19)]
print(agg(q.groupby('size')))
print('==== v3 tmean x size')
v3=v3.assign(tb2=pd.cut(v3.tmean,[-.1,1,2,3,4,5,6.1]))
print(agg(v3.groupby(['size','tb2'],observed=True)))
print(agg(v3.groupby(['p_n_full','p_bar'],observed=True)))
