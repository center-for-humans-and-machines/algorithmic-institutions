# v4: the games v3's final gp touched (seed 1000), paired with the zero manager's same games:
# per touched game, the first push (decision round, my size, number pushed, targets' window means) and the paired change in lead.
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
def sub(d,m): return {k:v[m] for k,v in d.items()}
def pools(d):
    C=d['C'].astype(float);P=d['P'].astype(float);G=d['G']
    own=(G==d['mine'][:,None,None])
    return ((1.6*C-P)*own).sum(1).mean(-1), ((1.6*C-P)*~own).sum(1).mean(-1), own.sum(1)
a=load(B+'pf-20261112-1-v2_sweep/final/per_round.parquet'); z=sub(a,a['point']=='pf-20261112-1-v2_gp')
b=load(B+'pf-20261112-1-v3_sweep/final/per_round.parquet'); x=sub(b,b['point']=='pf-20261112-1-v3_gp')
oz,tz,mz=pools(z); ox,tx,mx=pools(x)
own=(x['G']==x['mine'][:,None,None]); Pm=x['P']*own
rows=[]
for i in np.where(Pm.sum((1,2))>0)[0]:
    r=np.where(Pm[i].sum(0)>0)[0]; t0=r[0]
    tg=np.where(Pm[i][:,t0]>0)[0]
    w=slice(t0-3,t0+1)
    cm=x['C'][i][:,w].mean(1)
    # did targets leave at the decision? (group at t0+1)
    left=[(x['G'][i][j,t0+1]!=x['mine'][i]) for j in tg]
    rows.append(dict(i=i,dec=t0,n_dec=len(r),size=own[i][:,t0].sum(),npush=len(tg),tmean=cm[tg].mean(),
        left=np.mean(left), dpun=Pm[i].sum(), dlead=(ox[i]-tx[i])-(oz[i]-tz[i]), down=ox[i]-oz[i], doth=tx[i]-tz[i],
        dmem=mx[i].mean()-mz[i].mean(), seat=x['mine'][i]))
df=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(len(df), 'total dlead/2000 =', df.dlead.sum()/2000)
for k in ['dec','npush','n_dec','size']:
    print(df.groupby(k).agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem'),down=('down','mean'),doth=('doth','mean'),dmem=('dmem','mean'),left=('left','mean'),dpun=('dpun','mean')).round(2))
df['tbin']=pd.cut(df.tmean,[-0.1,0.5,1.5,3.1])
print(df.groupby('tbin',observed=True).agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem')).round(2))
print(df.groupby(['dec','npush']).agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem')).round(2))
df.to_csv('/tmp/v4_events.csv',index=False)
