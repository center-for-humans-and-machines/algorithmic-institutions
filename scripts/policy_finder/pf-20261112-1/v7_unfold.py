# v7: how the gain of v6's touched games unfolds over rounds against the zero twin: per-round own pool, ah's pool,
# own members; split by decision of the first push, by whether the target left at the next decision, and by size.
import numpy as np, pandas as pd
Z=pd.read_parquet('/tmp/v6_Z.parquet')
tw={}
def twin(ver,part):
    if (ver,part) not in tw:
        d=np.load(f'/tmp/v6_{ver}_{part}.npz',allow_pickle=True); tw[(ver,part)]={a:d[a] for a in ['por','ptr','G','seat','C']}
    return tw[(ver,part)]
rows=[]
for part in '12':
    d=np.load(f'/tmp/v6_v6_{part}.npz',allow_pickle=True)
    key,seat,pun,P,G,C,por,ptr=d['key'],d['seat'],d['pun'],d['P'],d['G'],d['C'],d['por'],d['ptr']
    idx=np.where(pun)[0]
    m=pd.DataFrame(dict(key=key[idx],seat=seat[idx],i=idx)).merge(Z,on=['key','seat'])
    for (ver,zp),g in m.groupby(['ver','part']):
        T=twin(ver,zp)
        for i,r in zip(g.i.values,g.row.values):
            pr=P[i]; rr=np.where(pr.sum(0)>0)[0][0]; tgt=np.where(pr[:,rr]>0)[0]
            own=(G[i]==seat[i]); ownz=(T['G'][r]==seat[i])
            left=int(own[tgt[0],min(rr+1,23)]==0)
            rows.append(dict(dec=rr,ntgt=len(tgt),size=own[:,rr].sum(),left=left,dpo=por[i]-T['por'][r],dpt=ptr[i]-T['ptr'][r],
                             dmem=own.sum(0)-ownz.sum(0),npun=(pr.sum(0)>0).sum()))
df=pd.DataFrame(rows); print(len(df))
for k in ['dpo','dpt','dmem']: df[k]=list(map(np.asarray,df[k]))
def show(g,name):
    po=np.stack(g.dpo.values).mean(0); pt=np.stack(g.dpt.values).mean(0); mm=np.stack(g.dmem.values).mean(0)
    print(name,'n',len(g),'lead/game %.2f'%(po-pt).mean())
    print('  dpo ',np.round(po,1)); print('  dpt ',np.round(pt,1)); print('  dmem',np.round(mm,2))
for dec,g in df.groupby('dec'): show(g,f'dec {dec}')
for (dec,l),g in df.groupby(['dec','left']): show(g,f'dec {dec} left {l}')
for s,g in df.groupby('size'): show(g,f'size {s}')
print(df.groupby(['dec','npun']).size())
df.drop(columns=['dpo','dpt','dmem']).assign(lead=[ (a-b).mean() for a,b in zip(df.dpo,df.dpt)]).to_parquet('/tmp/v7_unfold.parquet')
