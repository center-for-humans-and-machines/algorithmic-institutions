# v4: paired effects from the sweeps. Games at the same (point index, seat, episode) replay identically across
# v1/v2/v3 when unpunished, so every punished game has its zero-manager twin in another version.
# Writes one row per punished game: version, params, first punishment's round, my size, ah's size, n targets,
# targets' window mean and dose, total dealt, and the paired change in lead / own pool / ah pool / members.
import sys, numpy as np, json, pandas as pd
sys.path.insert(0,'/opt/algorithmic-institutions/scripts/policy_finder/pf-20261112-1')
from v3_load import load
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
V=['pf-20261112-1','pf-20261112-1-v2','pf-20261112-1-v3']
out=[]
for part in ['1','2']:
    res={}
    for v in V:
        d=load(B+f'{v}_sweep/per_round_p{part}of2.parquet')
        own=(d['G']==d['mine'][:,None,None])
        C=d['C'].astype(np.float32);P=d['P'].astype(np.float32)
        pr=(1.6*C-P)
        d['po']=(pr*own).sum(1); d['pt']=(pr*~own).sum(1)  # [eps,24]
        d['own']=own; d['Pm']=d['P']*own
        d['pun']=d['Pm'].sum((1,2))
        res[v]=d
    # zero twin: first version unpunished
    n=len(res[V[0]]['pun'])
    zpo=np.full((n,24),np.nan,np.float32); zpt=zpo.copy(); zmem=np.full(n,np.nan)
    for v in V:
        d=res[v]; m=(d['pun']==0)&np.isnan(zpo[:,0])
        zpo[m]=d['po'][m]; zpt[m]=d['pt'][m]; zmem[m]=d['own'][m].sum(1).mean(-1)
    print('part',part,'slots with zero twin',(~np.isnan(zmem)).mean())
    for v in V:
        d=res[v]; sw=json.load(open(B+f'{v}_sweep/sweep.json'))
        pars={p['name']:p['params'] for p in sw['points']}
        I=np.where((d['pun']>0)&~np.isnan(zmem))[0]
        Pm=d['Pm'][I]; own=d['own'][I]; C=d['C'][I]; G=d['G'][I]
        tot=Pm.sum(1)  # [k,24]
        t0=(tot>0).argmax(1)
        for j,i in enumerate(I):
            t=t0[j]; tg=np.where(Pm[j][:,t]>0)[0]; w0=(t//4)*4
            cm=C[j][:,w0:t+1].mean(1)
            nxt=min(t+1,23)
            row=dict(v=v[-2:] if v!=V[0] else 'v1', point=d['point'][i], seat=d['mine'][i], t0=t,
                     size=own[j][:,t].sum(), ntg=len(tg), tmean=cm[tg].mean(), dose=Pm[j][tg,t].mean(),
                     left=(G[j][tg,nxt]!=d['mine'][i]).mean() if t%4==3 and t<23 else np.nan,
                     total=Pm[j].sum(), nrounds=(tot[j]>0).sum(),
                     memmin=cm[own[j][:,t]].min(),
                     dlead=(d['po'][i]-d['pt'][i]).mean()-(zpo[i]-zpt[i]).mean(),
                     down=d['po'][i].mean()-zpo[i].mean(), doth=d['pt'][i].mean()-zpt[i].mean(),
                     dmem=own[j].sum(0).mean()-zmem[i])
            row.update({'p_'+k:x for k,x in pars[d['point'][i]].items()})
            out.append(row)
    del res
df=pd.DataFrame(out); df.to_parquet('/tmp/v4_paired.parquet')
print(df.groupby('v').agg(n=('dlead','size'),dlead=('dlead','mean')))
