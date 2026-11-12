# v6: touched games of v2..v5 against zero twins with the state at the first punishment: round, my size,
# number of targets, target(s)' window mean, others' window mean (ow), gap, dose, later punishment count.
import numpy as np, pandas as pd, json
Z=pd.read_parquet('/tmp/v6_Z.parquet')
B='/opt/algorithmic-institutions/plots/simulation/policy_finder/'
V={'v2':'pf-20261112-1-v2','v3':'pf-20261112-1-v3','v4':'pf-20261112-1-v4','v5':'pf-20261112-1-v5'}
out=[]
for k,v in V.items():
    pars={p['name']:p['params'] for p in json.load(open(B+v+'_sweep/sweep.json'))['points']}
    for part in '12':
        d=np.load(f'/tmp/v6_{k}_{part}.npz',allow_pickle=True)
        f=pd.DataFrame(dict(row=np.arange(len(d['key'])),key=d['key'],seat=d['seat'],point=d['point'],pun=d['pun'],lead=d['lead'],po=d['po'],pt=d['pt']))
        f=f[f.pun].merge(Z,on=['key','seat'],suffixes=('','_z'))
        C=d['C']; G=d['G']; Pm=d['P']; seat=d['seat']
        for r,dl,dpo,dpt,pt_ in zip(f.row,f.lead-f.lead_z,f.po-f.po_z,f.pt-f.pt_z,f.point):
            own=G[r]==seat[r]; tot=Pm[r].sum(0); pr=np.where(tot>0)[0]; t=int(pr[0]); w0=(t//4)*4
            tg=Pm[r][:,t]>0; mem=own[:,t]; cw=C[r][:,w0:t+1].astype(np.float32).mean(1); oth=mem&~tg
            out.append(dict(v=k,point=pt_,dlead=dl,dpo=dpo,dpt=dpt,t0=t,size=int(mem.sum()),ntg=int(tg.sum()),
                            tw=cw[tg].mean(),tmax=cw[tg].max(),ow=cw[oth].mean() if oth.any() else np.nan,dose=float(Pm[r][tg,t].mean()),
                            nround=len(pr),total=float(Pm[r].sum()),later=float(Pm[r][:,t+1:].sum()),
                            **{'p_'+a:b for a,b in pars[pt_].items()}))
        print(k,part,len(out),flush=True)
T=pd.DataFrame(out); T['gap']=T.ow-T.tw; T.to_parquet('/tmp/v6_all_touched.parquet')
