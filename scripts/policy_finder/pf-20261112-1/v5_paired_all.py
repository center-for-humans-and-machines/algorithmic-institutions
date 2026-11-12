# v5: paired effects for all four versions' sweeps keyed by (seed, slot): batch k of a part uses seed
# part_seed+k (1000 games per batch). Zero twin: the first of v1..v4 unpunished at that key with the same seat.
# Writes one row per punished game (all versions) with params and per-push detail to /tmp/v5_paired.parquet,
# and per-game arrays of the zero twin to /tmp/v5_zero.npz for later what-if analysis.
import sys, numpy as np, json, pandas as pd
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
V={'v1':'pf-20261112-1','v2':'pf-20261112-1-v2','v3':'pf-20261112-1-v3','v4':'pf-20261112-1-v4'}
data={}
for k,v in V.items():
    sw=json.load(open(B+f'{v}_sweep/sweep.json')); seeds=sw['seeds']
    pars={p['name']:p['params'] for p in sw['points']}
    for pi,part in enumerate(['1','2']):
        d=load(B+f'{v}_sweep/per_round_p{part}of2.parquet')
        n=len(d['C']); key=(seeds[pi]+np.arange(n)//1000)*1000+np.arange(n)%1000
        own=(d['G']==d['mine'][:,None,None])
        C=d['C'].astype(np.float32);P=d['P'].astype(np.float32)
        pr=1.6*C-P
        po=(pr*own).sum(1); pt=(pr*~own).sum(1)
        Pm=(d['P']*own)
        data[(k,part)]=dict(key=key,mine=d['mine'],point=d['point'],po=po,pt=pt,own=own,Pm=Pm,C=d['C'],G=d['G'],
                            pun=Pm.sum((1,2)),pars=pars)
        print(k,part,n,key.min()//1000,key.max()//1000)
# zero twin dict
Z={}
for k in V:
    for part in '12':
        d=data[(k,part)]
        for i in np.where(d['pun']==0)[0]:
            kk=(int(d['key'][i]),int(d['mine'][i]))
            if kk not in Z: Z[kk]=(d['po'][i],d['pt'][i],d['own'][i].sum(0).mean(),d['C'][i],d['G'][i])
print('zero twins',len(Z))
out=[]
for k in V:
    for part in '12':
        d=data[(k,part)]
        for i in np.where(d['pun']>0)[0]:
            kk=(int(d['key'][i]),int(d['mine'][i]))
            if kk not in Z: continue
            zpo,zpt,zm,zC,zG=Z[kk]
            Pm=d['Pm'][i]; own=d['own'][i]; C=d['C'][i]; G=d['G'][i]
            tot=Pm.sum(0); t=int((tot>0).argmax()); tg=np.where(Pm[:,t]>0)[0]; w0=(t//4)*4
            cm=C[:,w0:t+1].mean(1); nxt=min(t+1,23)
            decs=sorted(set((np.where(tot>0)[0]).tolist()))
            row=dict(v=k,point=d['point'][i],seat=int(d['mine'][i]),key=kk[0],t0=t,size=int(own[:,t].sum()),ntg=len(tg),
                     tmean=float(cm[tg].mean()),tmax=float(cm[tg].max()),dose=float(Pm[tg,t].mean()),
                     left=float((G[tg,nxt]!=d['mine'][i]).mean()) if t%4==3 and t<23 else np.nan,
                     total=float(Pm.sum()),ndec=len(decs),decs=str(decs),
                     dlead=float((d['po'][i]-d['pt'][i]).mean()-(zpo-zpt).mean()),
                     down=float(d['po'][i].mean()-zpo.mean()),doth=float(d['pt'][i].mean()-zpt.mean()),
                     dmem=float(own.sum(0).mean()-zm))
            row.update({'p_'+a:b for a,b in d['pars'][d['point'][i]].items()})
            out.append(row)
df=pd.DataFrame(out); df.to_parquet('/tmp/v5_paired.parquet')
print(df.groupby('v').agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem')))
# per point totals (games per point incl. unpunished)
cnt=[]
for k in V:
    for part in '12':
        d=data[(k,part)]
        u,c=np.unique(d['point'],return_counts=True); cnt+= [dict(v=k,point=a,games=b) for a,b in zip(u,c)]
pd.DataFrame(cnt).groupby(['v','point']).games.sum().reset_index().to_parquet('/tmp/v5_counts.parquet')
