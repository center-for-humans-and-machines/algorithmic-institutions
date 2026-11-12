# v5: per pushed target at the first push of a v4 game, his response against his zero twin: whether he left
# (game vs twin), his contribution over the next window in game and twin, and whether he returns later.
import sys, numpy as np, json, pandas as pd
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
pd.set_option('display.width',220)
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
V={'v1':'pf-20261112-1','v2':'pf-20261112-1-v2','v3':'pf-20261112-1-v3'}
Z={}
for k,v in V.items():
    sw=json.load(open(B+f'{v}_sweep/sweep.json')); seeds=sw['seeds']
    for pi,part in enumerate('12'):
        d=load(B+f'{v}_sweep/per_round_p{part}of2.parquet'); n=len(d['C'])
        key=(seeds[pi]+np.arange(n)//1000)*1000+np.arange(n)%1000
        pun=(d['P']*(d['G']==d['mine'][:,None,None])).sum((1,2))
        for i in np.where(pun==0)[0]:
            kk=(int(key[i]),int(d['mine'][i]))
            if kk not in Z: Z[kk]=(d['C'][i],d['G'][i])
        del d
sw=json.load(open(B+'pf-20261112-1-v4_sweep/sweep.json')); seeds=sw['seeds']; pars={p['name']:p['params'] for p in sw['points']}
rows=[]
for pi,part in enumerate('12'):
    d=load(B+f'pf-20261112-1-v4_sweep/per_round_p{part}of2.parquet'); n=len(d['C'])
    key=(seeds[pi]+np.arange(n)//1000)*1000+np.arange(n)%1000
    own=(d['G']==d['mine'][:,None,None]); Pm=d['P']*own
    for i in np.where(Pm.sum((1,2))>0)[0]:
        kk=(int(key[i]),int(d['mine'][i]))
        if kk not in Z: continue
        zC,zG=Z[kk]; C=d['C'][i]; G=d['G'][i]; m=d['mine'][i]
        tot=Pm[i].sum(0); t=int((tot>0).argmax()); tg=np.where(Pm[i][:,t]>0)[0]
        w0=(t//4)*4; cm=C[:,w0:t+1].mean(1)
        rank_low=np.argsort(np.where(own[i][:,t],cm,99))
        for j in tg:
            rows.append(dict(t0=t,size=int(own[i][:,t].sum()),ntg=len(tg),cm=cm[j],dose=int(Pm[i][j,t]),
                left=int(G[j,t+1]!=m), zleft=int(zG[j,t+1]!=m),
                c_next=C[j,t+1:t+5].mean(), zc_next=zC[j,t+1:t+5].mean(),
                inown_rest=(G[j,t+1:]==m).mean(), zinown_rest=(zG[j,t+1:]==m).mean(),
                c_rest_own=np.where(G[j,t+1:]==m,C[j,t+1:],0).sum()/(24-t-1),
                zc_rest_own=np.where(zG[j,t+1:]==m,zC[j,t+1:],0).sum()/(24-t-1),
                pnext_other=d['P'][i][j,t+1:t+5].mean()*(G[j,t+1]!=m),
                second=int(cm[j]>cm[rank_low[0]])))
    del d
r=pd.DataFrame(rows); r.to_parquet('/tmp/v5_targets.parquet')
r['cmb']=pd.cut(r.cm,[-.1,.5,2,4,6,8])
print(r.groupby('cmb',observed=True).agg(n=('cm','size'),left=('left','mean'),zleft=('zleft','mean'),c_next=('c_next','mean'),zc_next=('zc_next','mean'),
  inown=('inown_rest','mean'),zinown=('zinown_rest','mean'),crest=('c_rest_own','mean'),zcrest=('zc_rest_own','mean'),pother=('pnext_other','mean')).round(2))
s=r[r.left==0]
print('stayers'); print(s.groupby('cmb',observed=True).agg(n=('cm','size'),c_next=('c_next','mean'),zc_next=('zc_next','mean'),zleft=('zleft','mean')).round(2))
r['db']=pd.cut(r.dose,[13,18,23,30])
print(r.groupby(['cmb','db'],observed=True).agg(n=('cm','size'),left=('left','mean'),crest=('c_rest_own','mean'),zcrest=('zc_rest_own','mean')).round(2))
