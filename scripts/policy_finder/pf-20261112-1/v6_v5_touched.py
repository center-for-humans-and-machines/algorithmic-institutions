# v6: v5's touched games against their zero twins, by the state at the first push: decision, my size,
# target's window mean and this-round contribution, the others' window mean, the gap, ah's group (size, mean).
import numpy as np, pandas as pd
P=pd.read_parquet('/tmp/v6_v5_paired.parquet')
rows=[]
for part in '12':
    d=np.load(f'/tmp/v6_v5_{part}.npz',allow_pickle=True)
    C=d['C'].astype(np.float32); G=d['G']; Pm=d['P']; seat=d['seat']; V=d['V']
    sub=P[(P.part==part)&P.pun]
    for r,dl,dpo,dpt,dm,bar,am in zip(sub.row,sub.dlead,sub.dpo,sub.dpt,sub.dmem,sub.bar,sub.amount):
        own=G[r]==seat[r]; tot=Pm[r].sum(0); pr=np.where(tot>0)[0]; t=int(pr[0]); w0=(t//4)*4
        tg=int(np.where(Pm[r][:,t]>0)[0][0]); mem=own[:,t]; cw=C[r][:,w0:t+1].mean(1)
        oth=mem.copy(); oth[tg]=False; ah=~mem
        rows.append(dict(dlead=dl,dpo=dpo,dpt=dpt,dmem=dm,bar=bar,amount=am,t0=t,size=int(mem.sum()),npush=len(pr),
                         tw=cw[tg],tc=C[r][tg,t],ow=cw[oth].mean(),ahn=int(ah.sum()),ahw=cw[ah].mean() if ah.any() else np.nan,
                         tprev=C[r][tg,w0-4:w0].mean(), left=float(G[r][tg,t+1]!=seat[r])))
T=pd.DataFrame(rows); T.to_parquet('/tmp/v6_v5_touched.parquet')
def show(col,bins=None):
    x=pd.cut(T[col],bins) if bins is not None else T[col]
    print(T.groupby(x,observed=True).agg(n=('dlead','size'),dlead=('dlead','mean'),se=('dlead','sem'),dpo=('dpo','mean'),dpt=('dpt','mean'),left=('left','mean')).round(2))
print(len(T))
show('t0'); show('size'); show('npush'); show('tw',[-1,0,1,2,4,6,8,10,12]); show('tc',[-1,0,2,5,10,20])
T['gap']=T.ow-T.tw; show('gap',[-20,0,2,4,6,8,10,20]); show('ow',[0,6,8,10,12,14,20]); show('ahn'); show('ahw',[-1,0,2,5,10,20])
print(T.groupby(['t0','size'],observed=True).dlead.agg(['size','mean','sem']).round(2))
