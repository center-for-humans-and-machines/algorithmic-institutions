# v3: membership flows at the v2 final point: who leaves/joins my group, by contribution, punishment and opponent
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261106-11')
from d1_arrays import load
d = load('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v2_sweep/final/', 'v2final')
keep = d['pt']==-1  # only the surface peak (_gp); the final also re-tests other points
for k in ['c','p','g','v','mine','opp','run','pt','mine_g0']: d[k]=d[k][keep]
print('games', keep.sum(), 'points in final:', sorted(set(d['run'].tolist()))[:2])
c=d['c'].astype(float); p=d['p'].astype(float); m=d['mine']; opp=d['opp']
print('pun/head by round, mine:', np.round([(p[:,:,t]*m[:,:,t]).sum()/m[:,:,t].sum() for t in range(24)],2))
print('c by round mine:', np.round([(c[:,:,t]*m[:,:,t]).sum()/m[:,:,t].sum() for t in range(24)],1))
print('members by round per opponent')
for o in range(1,5):
    s = m[opp==o].sum(1).mean(0); so = (~m[opp==o]).sum(1).mean(0)
    print(o, np.round(s[[0,4,8,12,16,20,23]],2), 'opp group pun/head r0-3:', np.round([(p[opp==o][:,:,t]*~m[opp==o][:,:,t]).sum()/ (~m[opp==o][:,:,t]).sum() for t in [0,1,2,3,8,16]],2),
          'mine:', np.round([(p[opp==o][:,:,t]*m[opp==o][:,:,t]).sum()/ (m[opp==o][:,:,t]).sum() for t in [0,1,2,3,8,16]],2))
rows=[]
for t in [3,7,11,15,19]:
    blk = slice(t-3,t+1)
    for side in [True, False]:
        inb = m[:,:,t]==side
        leave = inb & (m[:,:,t+1]!=side)
        rows.append(pd.DataFrame(dict(t=t, mine=side, opp=np.repeat(opp,8), cmean=c[:,:,blk].mean(2).ravel(), pm=p[:,:,blk].mean(2).ravel(), pl=p[:,:,t].ravel(), leave=leave.ravel(), inb=inb.ravel())))
df=pd.concat(rows); df=df[df.inb]
df['cb']=pd.cut(df.cmean,[-1,4,8,12,16,20]); df['pb']=pd.cut(df.pm,[-1,0,2,5,10,30])
print('leave rate by block-mean c, mine vs opp group'); print(df.pivot_table(index='cb',columns=['mine'],values='leave',aggfunc='mean',observed=True).round(3))
print('leave rate (mine) by block mean pun and c'); print(df[df.mine].pivot_table(index='cb',columns='pb',values='leave',aggfunc='mean',observed=True).round(3))
print('leave rate per opponent (mine, other)'); print(df.pivot_table(index='opp',columns='mine',values='leave',aggfunc='mean').round(3))
# value flows: mean c of leavers vs joiners in my group, per opponent
df['cl']=df.cmean*df.leave
g=df.groupby(['opp','mine']).agg(n_leave=('leave','sum'), c_leave=('cl','sum'))
g['c_of_leavers']=g.c_leave/g.n_leave; print(g.round(2))
