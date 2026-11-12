# v6: at round 7 (no v5 push before it, so the zero manager's state) in v5's games: share of games by my size
# and by the gap (others' window mean minus the lowest member's) of the lowest valid member.
import numpy as np, pandas as pd
rows=[]
for part in '12':
    d=np.load(f'/tmp/v6_v5_{part}.npz',allow_pickle=True)
    C=d['C'][:,:,4:8].astype(np.float32); G=d['G'][:,:,7]; seat=d['seat']; V=d['V'][:,:,7]
    mem=(G==seat[:,None]); cw=C.mean(2)
    for i in range(len(seat)):
        m=mem[i]&(V[i]>0); n=int(mem[i].sum())
        if m.sum()<2: rows.append(dict(size=n,gap=np.nan,tw=np.nan)); continue
        x=cw[i][m]; j=x.argmin(); ow=(x.sum()-x[j])/(len(x)-1); rows.append(dict(size=n,gap=ow-x[j],tw=x[j]))
R=pd.DataFrame(rows)
print(pd.crosstab(R['size'],pd.cut(R.gap,[-1,4,6,8,10,12,30]),normalize='all').round(3))
