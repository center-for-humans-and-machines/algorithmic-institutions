# v7: first pushes of a sweep's (argv[1], default v6) touched games: target features at the push, whether he left (treated) and whether he
# left in the zero twin, and the paired lead. Features: c now, window mean, previous switches, rounds in my group,
# others' window mean (gap), size, dose, decision.
import numpy as np, pandas as pd, sys
VER=sys.argv[1] if len(sys.argv)>1 else 'v6'
Z=pd.read_parquet('/tmp/v6_Z.parquet')
tw={}
def twin(ver,part):
    if (ver,part) not in tw:
        d=np.load(f'/tmp/v6_{ver}_{part}.npz',allow_pickle=True); tw[(ver,part)]={a:d[a] for a in ['G','lead']}
    return tw[(ver,part)]
rows=[]
for part in '12':
    d=np.load(f'/tmp/v6_{VER}_{part}.npz',allow_pickle=True)
    key,seat,pun,P,G,C,V,lead=d['key'],d['seat'],d['pun'],d['P'],d['G'],d['C'],d['V'],d['lead']
    idx=np.where(pun)[0]
    m=pd.DataFrame(dict(key=key[idx],seat=seat[idx],i=idx)).merge(Z,on=['key','seat'])
    for (ver,zp),g in m.groupby(['ver','part']):
        T=twin(ver,zp)
        for i,r in zip(g.i.values,g.row.values):
            pr=P[i]; rr=np.where(pr.sum(0)>0)[0][0]; tg=np.where(pr[:,rr]>0)[0][0]; nt=(pr[:,rr]>0).sum()
            if rr<4 or rr>=23: continue
            own=(G[i]==seat[i]); ownz=(T['G'][r]==seat[i])
            w0=rr-3; cw=C[i][:,w0:rr+1].astype(float); vw=V[i][:,w0:rr+1]
            cm=(cw*vw).sum(1)/np.maximum(vw.sum(1),1)
            mem=own[:,rr]; oth=mem.copy(); oth[tg]=False
            sw=(np.diff(G[i][tg,:rr+1].astype(int))!=0).sum()
            rows.append(dict(dec=rr,size=mem.sum(),c_now=C[i][tg,rr],cm=cm[tg],ow=cm[oth].mean(),sw=sw,
                             since=rr-np.max(np.r_[-1,np.where(np.diff(G[i][tg,:rr+1].astype(int))!=0)[0]]),
                             dose=pr[tg,rr],left=int(not own[tg,rr+1]),leftz=int(not ownz[tg,rr+1]),
                             dlead=lead[i]-T['lead'][r],npun=(pr.sum(0)>0).sum(),nt=nt))
df=pd.DataFrame(rows); df['gap']=df.ow-df.cm
df.to_parquet(f'/tmp/v7_leave_{VER}.parquet'); print(len(df))
print('left %.3f leftz %.3f'%(df.left.mean(),df.leftz.mean()))
print(pd.crosstab(df.leftz,df.left,values=df.dlead,aggfunc='mean').round(2)); print(pd.crosstab(df.leftz,df.left))
for c,b in [('c_now',[-1,0,2,5,10,20]),('cm',[-1,0.5,2,4,6,9,12]),('sw',[-1,0,1,2,5]),('size',[5,6,7,8]),('gap',[-20,2,4,6,8,10,20]),('ow',[0,6,8,10,12,14,20])]:
    df['b']=pd.cut(df[c],b)
    print(df.groupby(['dec','b'],observed=True).agg(n=('left','size'),left=('left','mean'),leftz=('left','mean'),dlead=('dlead','mean'),se=('dlead','sem')).round(2).unstack(0) if False else
          df.groupby('b',observed=True).agg(n=('left','size'),left=('left','mean'),leftz=('leftz','mean'),dlead=('dlead','mean'),se=('dlead','sem')).round(2))
