# v5: v4's paired rows per touched game: by first decision, size, targets' window mean, dose, n targets, n decisions
import pandas as pd, numpy as np
pd.set_option('display.width',220)
df=pd.read_parquet('/tmp/v5_paired.parquet'); v=df[df.v=='v4'].copy()
def tab(k,bins=None):
    q=v.copy()
    if bins is not None: q[k]=pd.cut(q[k],bins)
    print(q.groupby(k,observed=True).agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem'),down=('down','mean'),doth=('doth','mean'),left=('left','mean'),dmem=('dmem','mean')).round(3))
tab('t0'); tab('size'); tab('tmean',[-.1,0.5,2,4,6,8]); tab('tmax',[-.1,0.5,2,4,6,8]); tab('dose',[13,16,18,20,22,24,26,28,30]); tab('ntg'); tab('ndec')
print(v.groupby(['t0','size']).dlead.agg(['size','mean','sem']).round(3))
print(v.groupby([pd.cut(v.tmax,[-.1,2,4,6,8]),'size']).dlead.agg(['size','mean','sem']).round(3))
